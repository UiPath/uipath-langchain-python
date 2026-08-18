"""Model-onboarding test case: run the file_processing agent on a model.

Runs the coded ``file_processing`` agent (Studio Web's "Clone as Coded Agent"
of the low-code FileProcessingAgent) against a model supplied at runtime via
``input.json``, once per file.

    {
      "model_spec": {
        "model_name": "gpt-5.2-2025-12-11",
        "api_flavors": ["openai:responses"],
        "agenthub_config": "agentsplayground"
      }
    }

Every file in ``FILE_REGISTRY`` is exercised. Each asks a question whose answer
is **unguessable** — a random code inside a PDF, the colour of a shape in an
image. That property is what makes the assertion meaningful: there is no prior
a model can fall back on, so producing the answer proves it read the file.

This matters more than it sounds. Earlier fixtures asked "what animal is in
this image?" over ``dog.jpg``; "dog" is the most likely answer to that question
with no image at all, and the file name was visible in the prompt, so a model
that never opened the file still scored correct.

Each ``api_flavors`` entry is a ``vendor_type:api_flavor`` pair forwarded to
``get_chat_model`` (e.g. ``openai:responses``, ``awsbedrock:converse``,
``vertexai:generate-content``); ``vendor_type:`` alone autodetects the flavor.
"""

import logging
import os
import re

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from uipath.llm_client.settings import PlatformSettings

from uipath_langchain.agent.multimodal.types import FileInfo
from uipath_langchain.chat.chat_model_factory import get_chat_model

from agents.file_processing.agent import run as run_file_processing
from agents.judge_guardrail.agent import run as run_judge_guardrail

logger = logging.getLogger(__name__)

class FileCase(BaseModel):
    """A file plus the question to ask and the answer that proves it was read.

    Each question has one deterministic word as its answer, and that word is
    absent from the file name — so an agent that never opened the file cannot
    produce it.
    """

    file: FileInfo
    question: str
    expected: str

    model_config = {"arbitrary_types_allowed": True}


# Files the agent processes. Both fixtures are generated and committed under
# fixtures/ (see fixtures/README.md), and both answers are unguessable — which
# is the whole point.
#
# The previous fixtures were borrowed from the web and both were guessable:
#
# - dog.jpg asked "what animal is this?", and "dog" is the single most likely
#   answer to that question with no image at all. A model that never opened the
#   file scored correct. The file name was visible in the prompt too.
# - dummy.pdf's text is the literal string "Dummy PDF file", which reads as a
#   placeholder, so the model kept editorializing about whether the content was
#   real instead of reporting it — flaky in both directions.
#
# A random code and an arbitrary color have no prior to fall back on: the model
# either read the file or it did not.
FILE_REGISTRY: dict[str, FileCase] = {
    "image": FileCase(
        # A purple square on white. The subject carries no colour prior.
        file=FileInfo(
            url="fixtures/shape.png",
            name="shape.png",
            mime_type="image/png",
        ),
        question=(
            "What colour is the large shape in the centre of this image? "
            "Answer with one word only."
        ),
        expected="purple",
    ),
    "pdf": FileCase(
        # A one-page PDF whose only text is "Verification code: PDF-CODE-74915".
        file=FileInfo(
            url="fixtures/document.pdf",
            name="document.pdf",
            mime_type="application/pdf",
        ),
        # "Answer with the code only" alone was not enough: one run replied
        # "Extracted verification code from document via Analyze_Files." —
        # describing the action instead of reporting the value. Naming the
        # failure mode explicitly is what stops it.
        question=(
            "What is the verification code written in this document? "
            "Reply with the code itself and nothing else. Do not describe "
            "what you did or which tool you used."
        ),
        expected="PDF-CODE-74915",
    ),
}


def _matches(expected: str, answer: str) -> bool:
    """Does the answer contain the expected value as a whole token?

    Deliberately simple, and only safe because the expected values are
    unguessable (a random code, an arbitrary colour). Earlier fixtures were
    guessable, which forced a refusal-phrase blocklist and a
    position-in-answer heuristic to tell "read the file" from "guessed the
    obvious"; both were brittle — they rejected correct answers that carried
    commentary, and still passed "there is no dog; it is a cat".

    Choosing a fixture whose answer cannot be guessed removes the need to
    interpret the prose around it: presence of the token *is* the evidence.
    Matching is case-insensitive and on whole tokens, so a substring like
    "Samoyed" cannot satisfy "dog", and markdown or a parser prefix around the
    answer is harmless.
    """
    # Hyphens are token separators here, so "PDF-CODE-74915" is compared as its
    # parts in order — robust to the model reformatting the separator.
    def tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    expected_tokens = tokens(expected)
    answer_tokens = tokens(answer)
    if not expected_tokens:
        return False
    # Look for the expected token sequence anywhere in the answer.
    span = len(expected_tokens)
    return any(
        answer_tokens[i : i + span] == expected_tokens
        for i in range(len(answer_tokens) - span + 1)
    )


class ModelSpec(BaseModel):
    """The model under test."""

    model_name: str = Field(description="Vendor-qualified model identifier.")
    api_flavors: list[str] = Field(
        description="'vendor_type:api_flavor' pairs forwarded to get_chat_model.",
    )
    agenthub_config: str = Field(default="agentsplayground")


class GraphInput(BaseModel):
    model_spec: ModelSpec


class GraphOutput(BaseModel):
    success: bool
    result_summary: str


def _one_line(error: BaseException) -> str:
    """Collapse an exception to one line for the per-cell summary."""
    return " ".join(str(error).split())


async def probe_file_processing(state: GraphInput) -> GraphOutput:
    """Run the agent for every api_flavor x file, collecting its answers.

    Nothing here is allowed to abort the run. Every cell — settings, each
    model build, each file — is recorded as ✓ or ✗ and the loop continues,
    because a model that fails one flavor is exactly the condition this test
    is meant to *report*. An unguarded raise kills the node, `uipath run`
    exits non-zero, and `set -e` in run.sh stops the script before
    validate_output.sh, discarding the summary and every flavor that already
    passed.
    """
    spec = state.model_spec
    lines: list[str] = []
    failed = False

    def record(line: str) -> None:
        """Append a summary line and log it immediately.

        Logged as it happens so a run that dies for an unforeseen reason still
        leaves the partial results in the job log.
        """
        lines.append(line)
        logger.info(line)

    try:
        settings = PlatformSettings(agenthub_config=spec.agenthub_config)
    except Exception as e:
        # A bad agenthub_config fails every flavor, so report it once and stop
        # rather than repeat the same error per flavor.
        detail = f"settings: ✗ {type(e).__name__}: {_one_line(e)}"[:300]
        record(detail)
        return GraphOutput(success=False, result_summary="\n".join(lines))

    for flavor in spec.api_flavors:
        record(f"{flavor}:")
        vendor_type, _, api_flavor = flavor.partition(":")

        # get_chat_model raises on an unknown vendor (ValueError) or a model
        # the tenant cannot serve (AgentStartupError, via ModelNotFoundError).
        # Both are ordinary findings for this test, not crashes.
        try:
            model = get_chat_model(
                model=spec.model_name,
                client_settings=settings,
                vendor_type=vendor_type or None,
                api_flavor=api_flavor or None,
                temperature=0.0,
                max_tokens=2000,
            )
        except Exception as e:
            failed = True
            record(f"  build: ✗ {type(e).__name__}: {_one_line(e)}"[:300])
            continue

        # Log the class actually constructed: discovery can override the
        # requested api_flavor, and an unrecognized Bedrock flavor falls
        # through to Converse — so echoing the requested string alone would
        # sign off on a surface that was never exercised.
        record(f"  build: ✓ {type(model).__name__}")

        for file_name, case in FILE_REGISTRY.items():
            try:
                answer = await run_file_processing(
                    model, case.question, [case.file]
                )
            except Exception as e:
                failed = True
                record(f"  {file_name}: ✗ {type(e).__name__}: {_one_line(e)}"[:300])
                continue

            if _matches(case.expected, answer):
                record(f"  {file_name}: ✓ {answer}"[:200])
            else:
                failed = True
                record(
                    f"  {file_name}: ✗ expected '{case.expected}', got: {answer}"[:300]
                )

        # The model under test in the *judge* role, behind a real LLM-as-judge
        # guardrail. Skipped without a user-identity token: the validator lives
        # on `agentsruntime_`, which the client-credentials app cannot reach —
        # the OAuth resource catalog has no `agentsruntime` entry, so an S2S
        # token comes back 401. Skipping rather than failing keeps the default
        # CI path (S2S secrets, no PAT) green; see run.sh for the PAT wiring.
        if not os.environ.get("UIPATH_PAT"):
            record("  judge_guardrail: – skipped (no UIPATH_PAT; needs user identity)")
            continue

        try:
            record(f"  judge_guardrail: ✓ {await run_judge_guardrail(model, spec.model_name)}"[:300])
        except Exception as e:
            failed = True
            record(f"  judge_guardrail: ✗ {type(e).__name__}: {_one_line(e)}"[:300])

    if not spec.api_flavors:
        failed = True
        record("✗ no api_flavors supplied")

    summary = "\n".join(lines)
    logger.info(f"Success: {not failed}\nSummary:\n{summary}")
    return GraphOutput(success=not failed, result_summary=summary)


# `langgraph.json` points the runtime at ./src/main.py:graph, which must be a
# StateGraph; the runtime compiles it with its own checkpointer. The agent
# under test brings its own graph — see agents/file_processing/agent.py.
graph = StateGraph(GraphInput, output_schema=GraphOutput)
graph.add_node(probe_file_processing)
graph.add_edge(START, "probe_file_processing")
graph.add_edge("probe_file_processing", END)
