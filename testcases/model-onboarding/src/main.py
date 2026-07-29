"""Model-onboarding test case: run the file_processing agent on a model.

Runs the coded ``file_processing`` agent (Studio Web's "Clone as Coded Agent"
of the low-code FileProcessingAgent) against a model supplied at runtime via
``input.json``, once per file, and returns the model's own answers.

    {
      "prompt": "Describe the content of this file in one sentence.",
      "model_spec": {
        "model_name": "gpt-5.2-2025-12-11",
        "api_flavors": ["openai:responses"],
        "agenthub_config": "agentsplayground",
        "files": ["image", "pdf"]
      }
    }

Each ``api_flavors`` entry is a ``vendor_type:api_flavor`` pair forwarded to
``get_chat_model`` (e.g. ``openai:responses``, ``awsbedrock:converse``,
``vertexai:generate-content``); ``vendor_type:`` alone autodetects the flavor.
"""

import logging

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from uipath.llm_client.settings import PlatformSettings

from uipath_langchain.agent.multimodal.types import FileInfo
from uipath_langchain.chat.chat_model_factory import get_chat_model

from agents.file_processing.agent import run as run_file_processing

logger = logging.getLogger(__name__)

# Files the agent processes, selected by name via `model_spec.files`.
FILE_REGISTRY: dict[str, FileInfo] = {
    "image": FileInfo(
        url="https://www.w3schools.com/css/img_5terre.jpg",
        name="img_5terre.jpg",
        mime_type="image/jpeg",
    ),
    "pdf": FileInfo(
        url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        name="dummy.pdf",
        mime_type="application/pdf",
    ),
}

# Words the answer must contain to prove the agent read the file rather than
# guessing from its name/MIME type. At least one alternative per group must
# appear (case-insensitive).
FILE_EVIDENCE: dict[str, list[list[str]]] = {
    # Only visible in the photo: cliffside houses above the sea.
    "image": [["village", "houses", "buildings", "town"], ["sea", "water", "coast", "bay", "ocean"]],
    # The PDF's only content is the line "Dummy PDF file".
    "pdf": [["dummy"]],
}


def _missing_evidence(file_name: str, answer: str) -> str:
    """Return the first evidence group the answer fails to mention, if any.

    Guards against a model answering plausibly from the file name alone: the
    reply has to contain something only the file's contents reveal.
    """
    lowered = answer.lower()
    for group in FILE_EVIDENCE.get(file_name, []):
        if not any(word in lowered for word in group):
            return "/".join(group)
    return ""


class ModelSpec(BaseModel):
    """The model under test."""

    model_name: str = Field(description="Vendor-qualified model identifier.")
    api_flavors: list[str] = Field(
        description="'vendor_type:api_flavor' pairs forwarded to get_chat_model.",
    )
    agenthub_config: str = Field(default="agentsplayground")
    files: list[str] = Field(default_factory=lambda: ["image", "pdf"])


class GraphInput(BaseModel):
    prompt: str = Field(default="Describe the content of this file in one sentence.")
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

        for file_name in spec.files:
            try:
                answer = await run_file_processing(
                    model, state.prompt, [FILE_REGISTRY[file_name]]
                )
            except Exception as e:
                failed = True
                # Collapse newlines: some SDK errors are multi-line and would
                # break the one-cell-per-line summary.
                detail = " ".join(str(e).split())
                lines.append(f"  {file_name}: ✗ {type(e).__name__}: {detail}"[:300])
                continue

            missing = _missing_evidence(file_name, answer)
            if missing:
                failed = True
                lines.append(
                    f"  {file_name}: ✗ answer lacks {missing} (did the agent "
                    f"read the file?): {answer}"[:300]
                )
            else:
                lines.append(f"  {file_name}: ✓ {answer}"[:400])

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
