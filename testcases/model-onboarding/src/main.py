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
                answer = f"✗ {type(e).__name__}: {e}"
            lines.append(f"  {file_name}: {answer}")

    summary = "\n".join(lines)
    logger.info(f"Success: {not failed}\nSummary:\n{summary}")
    return GraphOutput(success=not failed, result_summary=summary)


# `langgraph.json` points the runtime at ./src/main.py:graph, so the probe is
# wrapped in a single-node graph. The agent under test brings its own graph;
# see agents/file_processing/agent.py.
def build_graph():
    builder = StateGraph(GraphInput, output_schema=GraphOutput)
    builder.add_node("probe_file_processing", probe_file_processing)
    builder.add_edge(START, "probe_file_processing")
    builder.add_edge("probe_file_processing", END)
    return builder.compile()


graph = build_graph()
