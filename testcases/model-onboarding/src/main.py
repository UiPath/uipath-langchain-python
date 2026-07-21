"""Model-parameterized onboarding test case.

Exercises a single model — supplied at runtime via ``input.json`` — across the
distinct ``get_chat_model`` code paths that model is expected to support, plus
an optional set of file attachments. Every ``path x file`` cell is invoked
independently; failures are caught per cell and rolled up into one ``success``
boolean, mirroring the ``multimodal-invoke`` contract so this project drops into
the existing integration-test matrix unchanged.

The model is NOT hardcoded. Edit ``input.json`` to onboard any model:

    {
      "prompt": "Describe the content of this file in one sentence.",
      "model_spec": {
        "model_name": "gpt-5.2-2025-12-11",
        "paths": ["azure_responses", "azure_chat_completions"],
        "agenthub_config": "agentsplayground",
        "files": ["image", "pdf"]
      }
    }

``paths`` is explicit on purpose: a model ID is only valid on the vendor
families it actually ships on, so the caller declares which surfaces to test
rather than the code guessing from the name. ``files`` may be empty for
text-only models — an empty list runs a plain ``ainvoke`` reachability check.
"""

import logging
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field
from uipath.llm_client.settings import PlatformSettings
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
)
from uipath_langchain_client.settings import (
    ApiFlavor,
    UiPathBaseSettings,
)

from uipath_langchain.agent.multimodal.invoke import llm_call_with_files
from uipath_langchain.agent.multimodal.types import FileInfo
from uipath_langchain.chat.chat_model_factory import get_chat_model

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Path registry: one builder per distinct get_chat_model code path.
#
# Each builder returns a configured BaseChatModel for the given model name and
# client settings. The keys are the strings the caller lists in
# `model_spec.paths`. Adding a new reachable class is a one-line addition here.
# --------------------------------------------------------------------------- #
PathBuilder = Callable[[str, UiPathBaseSettings], object]

PATH_REGISTRY: dict[str, PathBuilder] = {
    # VendorType.OPENAI (UiPath-owned) -> UiPathAzureChatOpenAI (responses API)
    "azure_responses": lambda model, settings: get_chat_model(
        model=model,
        client_settings=settings,
        api_flavor=ApiFlavor.RESPONSES,
        temperature=0.0,
        max_tokens=200,
    ),
    # VendorType.OPENAI + CHAT_COMPLETIONS -> UiPathAzureChatOpenAI (chat API)
    "azure_chat_completions": lambda model, settings: get_chat_model(
        model=model,
        client_settings=settings,
        api_flavor=ApiFlavor.CHAT_COMPLETIONS,
        temperature=0.0,
        max_tokens=200,
    ),
    # VendorType.VERTEXAI (Google family) -> UiPathChatGoogleGenerativeAI
    "vertex": lambda model, settings: get_chat_model(
        model=model,
        client_settings=settings,
        temperature=0.0,
        max_tokens=200,
    ),
    # VendorType.AWSBEDROCK (UiPath-owned) -> UiPathChatBedrockConverse
    "bedrock_converse": lambda model, settings: get_chat_model(
        model=model,
        client_settings=settings,
        api_flavor=ApiFlavor.CONVERSE,
        temperature=0.0,
        max_tokens=200,
    ),
    # VendorType.AWSBEDROCK + INVOKE -> UiPathChatBedrock
    "bedrock_invoke": lambda model, settings: get_chat_model(
        model=model,
        client_settings=settings,
        api_flavor=ApiFlavor.INVOKE,
        temperature=0.0,
        max_tokens=200,
    ),
    # Direct instantiation -> UiPathChatAnthropicBedrock (not factory-reachable)
    "anthropic_sdk": lambda model, settings: UiPathChatAnthropicBedrock(
        model_name=model,
        settings=settings,
        temperature=0.0,
        max_tokens=200,
    ),
}


# --------------------------------------------------------------------------- #
# File registry: named file attachments the caller selects via
# `model_spec.files`. Public, reachable URLs so the run environment can fetch
# them. Extend as needed for other formats you want to onboard against.
# --------------------------------------------------------------------------- #
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
    paths: list[str] = Field(
        description="get_chat_model code paths to exercise; keys of PATH_REGISTRY.",
    )
    agenthub_config: str = Field(
        default="agentsplayground",
        description="AgentHub config header value; must exist in the target tenant.",
    )
    files: list[str] = Field(
        default_factory=list,
        description="File attachments to test; keys of FILE_REGISTRY. Empty => "
        "text-only reachability check via plain ainvoke.",
    )


class GraphInput(BaseModel):
    prompt: str = Field(default="Describe the content of this file in one sentence.")
    model_spec: ModelSpec


class GraphOutput(BaseModel):
    success: bool
    result_summary: str


class GraphState(MessagesState):
    prompt: str
    model_spec: dict
    success: bool
    result_summary: str
    model_results: dict


def _build_model(path: str, model_name: str, settings: UiPathBaseSettings) -> object:
    """Build a model instance for a registered path.

    Raises:
        KeyError: If ``path`` is not a registered PATH_REGISTRY key.
    """
    builder = PATH_REGISTRY[path]
    return builder(model_name, settings)


async def run_model_onboarding(state: GraphState) -> dict:
    spec = ModelSpec.model_validate(state["model_spec"])
    messages = [HumanMessage(content=state["prompt"])]

    # Empty files => a single text-only cell (llm_call_with_files does a plain
    # ainvoke when the file list is empty).
    selected_files: list[tuple[str, list[FileInfo]]]
    if spec.files:
        selected_files = [(name, [FILE_REGISTRY[name]]) for name in spec.files]
    else:
        selected_files = [("text-only", [])]

    try:
        client_settings = PlatformSettings(agenthub_config=spec.agenthub_config)
    except Exception as e:
        # Settings need UiPath auth env vars (set by `uipath auth`). If they are
        # missing the whole run is moot; surface it as a legible failure.
        logger.error(f"PlatformSettings construction failed: {e}")
        return {
            "success": False,
            "result_summary": f"settings: ✗ {str(e)[:120]}",
            "model_results": {},
        }

    model_results: dict[str, dict[str, str]] = {}
    for path in spec.paths:
        logger.info(f"Testing path '{path}' with model '{spec.model_name}'...")

        if path not in PATH_REGISTRY:
            logger.error(f"  unknown path '{path}'")
            model_results[path] = {"__path__": f"✗ unknown path '{path}'"}
            continue

        try:
            model = _build_model(path, spec.model_name, client_settings)
            logger.info(f"  Created: {type(model).__name__}")
        except Exception as e:  # model construction itself can fail
            logger.error(f"  construction failed: {e}")
            model_results[path] = {"__build__": f"✗ {str(e)[:60]}"}
            continue

        cell_results: dict[str, str] = {}
        for label, files in selected_files:
            logger.info(f"  {label}...")
            try:
                response: AIMessage = await llm_call_with_files(
                    messages, files, model
                )
                # Guard against an empty/blank completion counting as success.
                if response.content and str(response.content).strip():
                    logger.info(f"    {label}: ✓")
                    cell_results[label] = "✓"
                else:
                    logger.warning(f"    {label}: ✗ empty response")
                    cell_results[label] = "✗ empty response"
            except Exception as e:
                logger.error(f"    {label}: ✗ {e}")
                cell_results[label] = f"✗ {str(e)[:60]}"
        model_results[path] = cell_results

    summary_lines = []
    for path, results in model_results.items():
        summary_lines.append(f"{path}:")
        for cell_name, result in results.items():
            summary_lines.append(f"  {cell_name}: {result}")

    has_failures = any(
        "✗" in v for results in model_results.values() for v in results.values()
    )
    # A spec with no runnable paths is a failure, not a vacuous success.
    if not model_results:
        has_failures = True
        summary_lines.append("(no paths specified)")

    return {
        "success": not has_failures,
        "result_summary": "\n".join(summary_lines),
        "model_results": model_results,
    }


async def return_results(state: GraphState) -> GraphOutput:
    logger.info(f"Success: {state['success']}")
    logger.info(f"Summary:\n{state['result_summary']}")
    return GraphOutput(
        success=state["success"],
        result_summary=state["result_summary"],
    )


def build_graph() -> StateGraph:
    builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)

    builder.add_node("run_model_onboarding", run_model_onboarding)
    builder.add_node("results", return_results)

    builder.add_edge(START, "run_model_onboarding")
    builder.add_edge("run_model_onboarding", "results")
    builder.add_edge("results", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()
