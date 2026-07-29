"""Model-onboarding test case: file processing.

Runs the coded ``file_processing`` agent (Studio Web's "Clone as Coded Agent"
of the low-code FileProcessingAgent) against a model supplied at runtime via
``input.json`` — one cell per attached file, rolled up into a single
``success`` boolean plus a ``result_summary`` carrying the model's actual
answer for each file.

    {
      "prompt": "Describe the content of this file in one sentence.",
      "model_spec": {
        "model_name": "gpt-5.2-2025-12-11",
        "api_flavors": ["azure_responses"],
        "agenthub_config": "agentsplayground",
        "files": ["image", "pdf"]
      }
    }

``api_flavors`` entries are either shorthands (``azure_responses``,
``azure_chat_completions``, ``vertex``, ``bedrock_converse``,
``bedrock_invoke``) or ``vendor_type:api_flavor`` pairs passed straight to
``get_chat_model`` (e.g. ``awsbedrock:converse``, ``openai:responses``).
"""

import logging

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from uipath.llm_client.settings import PlatformSettings
from uipath_langchain_client.settings import ApiFlavor, UiPathBaseSettings

from uipath_langchain.agent.multimodal.types import FileInfo
from uipath_langchain.chat.chat_model_factory import get_chat_model

from agents.file_processing.agent import run as run_file_processing

logger = logging.getLogger(__name__)

# Shorthand names for common vendor/api_flavor combinations. Anything not
# listed here is parsed as "vendor_type:api_flavor".
FLAVOR_SHORTHANDS: dict[str, tuple[str | None, ApiFlavor | None]] = {
    "azure_responses": (None, ApiFlavor.RESPONSES),
    "azure_chat_completions": (None, ApiFlavor.CHAT_COMPLETIONS),
    "vertex": ("vertexai", ApiFlavor.GENERATE_CONTENT),
    "bedrock_converse": (None, ApiFlavor.CONVERSE),
    "bedrock_invoke": (None, ApiFlavor.INVOKE),
}

# Files the agent is asked to process; selected by name via `model_spec.files`.
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
    """Runtime specification for the model under test."""

    model_name: str = Field(description="Vendor-qualified model identifier.")
    api_flavors: list[str] = Field(
        description="API flavors to exercise: FLAVOR_SHORTHANDS keys or "
        "'vendor_type:api_flavor' pairs.",
    )
    agenthub_config: str = Field(
        default="agentsplayground",
        description="AgentHub config header value; must exist in the target tenant.",
    )
    files: list[str] = Field(
        default_factory=lambda: ["image", "pdf"],
        description="Files to process; keys of FILE_REGISTRY.",
    )


class GraphInput(BaseModel):
    prompt: str = Field(default="Describe the content of this file in one sentence.")
    model_spec: ModelSpec


class GraphOutput(BaseModel):
    success: bool
    result_summary: str


class GraphState(TypedDict, total=False):
    prompt: str
    model_spec: dict
    success: bool
    result_summary: str


def build_model(flavor: str, model_name: str, settings: UiPathBaseSettings) -> object:
    """Build the chat model for one API flavor.

    Args:
        flavor: A FLAVOR_SHORTHANDS key, or ``vendor_type:api_flavor``
            (``vendor_type:`` alone lets the factory autodetect the flavor).
        model_name: Vendor-qualified model identifier.
        settings: Client settings carrying the AgentHub config.

    Returns:
        A configured chat model.

    Raises:
        ValueError: If ``flavor`` is neither a shorthand nor a vendor:flavor pair.
    """
    if flavor in FLAVOR_SHORTHANDS:
        vendor_type, api_flavor = FLAVOR_SHORTHANDS[flavor]
    elif ":" in flavor:
        vendor_raw, _, flavor_raw = flavor.partition(":")
        vendor_type = vendor_raw.strip() or None
        api_flavor = flavor_raw.strip() or None  # type: ignore[assignment]
    else:
        raise ValueError(
            f"unknown api_flavor '{flavor}': expected one of "
            f"{sorted(FLAVOR_SHORTHANDS)} or "
            "'vendor_type:api_flavor'"
        )

    return get_chat_model(
        model=model_name,
        client_settings=settings,
        vendor_type=vendor_type,
        api_flavor=api_flavor,
        temperature=0.0,
        max_tokens=2000,
    )


async def probe_file_processing(state: GraphState) -> dict:
    """Run the file_processing agent for every api_flavor x file combination."""
    spec = ModelSpec.model_validate(state["model_spec"])

    try:
        settings = PlatformSettings(agenthub_config=spec.agenthub_config)
    except Exception as e:
        # Settings need the auth env vars that `uipath auth` writes.
        logger.error(f"PlatformSettings construction failed: {e}")
        return {
            "success": False,
            "result_summary": f"settings: ✗ {type(e).__name__}: {e}"[:220],
        }

    lines: list[str] = []
    failed = False

    for flavor in spec.api_flavors:
        lines.append(f"{flavor}:")
        try:
            model = build_model(flavor, spec.model_name, settings)
            logger.info(f"{flavor}: built {type(model).__name__}")
        except Exception as e:
            failed = True
            lines.append(f"  build: ✗ {type(e).__name__}: {e}"[:220])
            continue

        for file_name in spec.files:
            file_info = FILE_REGISTRY.get(file_name)
            if file_info is None:
                failed = True
                lines.append(f"  {file_name}: ✗ unknown file")
                continue
            try:
                answer = await run_file_processing(model, state["prompt"], [file_info])
            except Exception as e:
                failed = True
                lines.append(f"  {file_name}: ✗ {type(e).__name__}: {e}"[:220])
                continue
            # The agent's own answer is the evidence the probe returns.
            lines.append(f"  {file_name}: ✓ {answer}"[:400])

    if not lines:
        failed = True
        lines.append("(no api_flavors specified)")

    return {"success": not failed, "result_summary": "\n".join(lines)}


async def return_results(state: GraphState) -> GraphOutput:
    logger.info(f"Success: {state['success']}")
    logger.info(f"Summary:\n{state['result_summary']}")
    return GraphOutput(
        success=state["success"], result_summary=state["result_summary"]
    )


def build_graph():
    builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)
    builder.add_node("probe_file_processing", probe_file_processing)
    builder.add_node("results", return_results)
    builder.add_edge(START, "probe_file_processing")
    builder.add_edge("probe_file_processing", "results")
    builder.add_edge("results", END)
    return builder.compile()


graph = build_graph()
