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

Every file in ``FILE_REGISTRY`` is exercised. Each asks a question with one
deterministic answer that only its contents reveal — "what animal is this?"
over a photo of a dog, "what is the first word inside?" over a PDF reading
"Dummy PDF file". The answer word appears nowhere in the file name, so a model
that never opened the file cannot produce it.

Each ``api_flavors`` entry is a ``vendor_type:api_flavor`` pair forwarded to
``get_chat_model`` (e.g. ``openai:responses``, ``awsbedrock:converse``,
``vertexai:generate-content``); ``vendor_type:`` alone autodetects the flavor.
"""

import logging
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


# Files the agent processes, selected by name via `model_spec.files`.
FILE_REGISTRY: dict[str, FileCase] = {
    "image": FileCase(
        # A white Samoyed sitting on grass.
        file=FileInfo(
            url="https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            name="animal.jpg",
            mime_type="image/jpeg",
        ),
        question="What animal is in this image? Answer with one word only.",
        expected="dog",
    ),
    "pdf": FileCase(
        # A one-page PDF whose entire content is the line "Dummy PDF file".
        file=FileInfo(
            url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            name="document.pdf",
            mime_type="application/pdf",
        ),
        # This file's text is the literal string "Dummy PDF file", which reads
        # as a placeholder — across six runs the model answered it correctly
        # three times and three times refused, reporting the file unreadable
        # while quoting the very text the tool had returned. Asking it to
        # repeat the tool output verbatim removes the judgement call; the
        # instruction not to evaluate the text is what makes this stable.
        question=(
            "Call the Analyze Files tool, then repeat its result back "
            "verbatim as your entire answer. Do not evaluate, judge or "
            "comment on whether the text is meaningful."
        ),
        expected="dummy",
    ),
}


# Phrases a model uses when it denies having read the file. Only the outright
# refusals belong here.
#
# Descriptive words like "placeholder" and "appears to be" were once on this
# list and had to come off: the model reads the PDF correctly, then volunteers
# commentary ("Content type: Plain text placeholder"). That is a *correct*
# answer with editorializing attached, and matching the word alone failed it.
_REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "unable",
    "not available",
    "unreadable",
)


def _matches(expected: str, answer: str) -> bool:
    """Was the expected word actually given as the answer?

    Requires the word as a real token and no refusal language. Length is not a
    criterion: a verbatim transcription legitimately carries the parser's
    ``<PARSED TEXT FOR PAGE: 1 / 1>`` prefix, which a word-count limit rejected
    even though the answer was correct.
    """
    lowered = answer.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return False
    # Split on non-word characters rather than stripping a hand-listed set of
    # punctuation: the model wraps the answer in markdown (`` `Dummy PDF file` ``),
    # and a strip-list missed the backtick, failing a correct answer.
    words = re.findall(r"\w+", lowered)
    return expected.lower() in words


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


async def probe_file_processing(state: GraphInput) -> GraphOutput:
    """Run the agent for every api_flavor x file, collecting its answers."""
    spec = state.model_spec
    settings = PlatformSettings(agenthub_config=spec.agenthub_config)
    lines: list[str] = []
    failed = False

    for flavor in spec.api_flavors:
        vendor_type, _, api_flavor = flavor.partition(":")
        model = get_chat_model(
            model=spec.model_name,
            client_settings=settings,
            vendor_type=vendor_type or None,
            api_flavor=api_flavor or None,
            temperature=0.0,
            max_tokens=2000,
        )
        lines.append(f"{flavor}:")

        for file_name, case in FILE_REGISTRY.items():
            try:
                answer = await run_file_processing(
                    model, case.question, [case.file]
                )
            except Exception as e:
                failed = True
                # Collapse newlines: some SDK errors are multi-line and would
                # break the one-cell-per-line summary.
                detail = " ".join(str(e).split())
                lines.append(f"  {file_name}: ✗ {type(e).__name__}: {detail}"[:300])
                continue

            if _matches(case.expected, answer):
                lines.append(f"  {file_name}: ✓ {answer}"[:200])
            else:
                failed = True
                lines.append(
                    f"  {file_name}: ✗ expected '{case.expected}', got: {answer}"[:300]
                )

        # LLM-as-judge guardrail with the model under test in the judge role.
        try:
            verdict = await run_judge_guardrail(model, spec.model_name)
            lines.append(f"  judge_guardrail: ✓ {verdict}"[:220])
        except Exception as e:
            failed = True
            detail = " ".join(str(e).split())
            lines.append(
                f"  judge_guardrail: ✗ {type(e).__name__}: {detail}"[:300]
            )

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
