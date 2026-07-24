"""Model-parameterized onboarding test case.

Exercises a single model — supplied at runtime via ``input.json`` — across the
distinct ``get_chat_model`` code paths that model is expected to support. For
every path, three capability payloads are run:

- ``simple``  — a plain text ``ainvoke``; asserts a non-empty completion.
- ``tools``   — a full tool-calling round trip: bind a tool, let the model
                request it, execute the tool, feed the ``ToolMessage`` back, and
                assert the model produces a final answer that uses the result.
- ``files/*`` — one cell per selected file attachment via ``llm_call_with_files``.

Every ``path x payload`` cell is invoked independently; failures are caught per
cell and rolled up into one ``success`` boolean, mirroring the
``multimodal-invoke`` output contract so this project drops into the existing
integration-test matrix unchanged.

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
families it actually ships on, so the caller declares which surfaces to test.
``files`` may be empty — the ``simple`` and ``tools`` payloads still run, only
the per-file cells are skipped.
"""

import logging
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
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


# --------------------------------------------------------------------------- #
# Tool-calling payload: a single deterministic tool. The prompt is written to
# force a call, and the expected answer is deterministic so the round trip can
# be asserted end to end.
# --------------------------------------------------------------------------- #
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city to look up the weather for.
    """
    # Deterministic so the final-answer assertion is stable.
    return f"The weather in {city} is 22 degrees Celsius and sunny."


TOOLS = [get_weather]
TOOL_PROMPT = "What is the weather in Paris? Use the get_weather tool."
# A token the final answer must contain to prove the tool result was consumed.
TOOL_ANSWER_TOKEN = "22"


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
        description="File attachments for the 'files' payload; keys of "
        "FILE_REGISTRY. Empty => only simple + tools payloads run.",
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


async def _run_simple(model: BaseChatModel, prompt: str) -> str:
    """Simple-call payload: plain ainvoke, require a non-empty completion."""
    response = await model.ainvoke([HumanMessage(content=prompt)])
    if not isinstance(response, AIMessage):
        return f"✗ non-AIMessage: {type(response).__name__}"
    if response.content and str(response.content).strip():
        return "✓"
    return "✗ empty response"


async def _run_tools(model: BaseChatModel) -> str:
    """Tool-calling payload: a full round trip.

    Binds the tool, forces a call, executes the tool locally, feeds the
    ToolMessage back, and asserts the final answer reflects the tool result.
    """
    llm = model.bind_tools(TOOLS)
    messages: list = [HumanMessage(content=TOOL_PROMPT)]

    first = await llm.ainvoke(messages)
    if not isinstance(first, AIMessage) or not first.tool_calls:
        return "✗ no tool call requested"

    messages.append(first)
    by_name = {t.name: t for t in TOOLS}
    for call in first.tool_calls:
        tool_obj = by_name.get(call["name"])
        if tool_obj is None:
            return f"✗ unexpected tool '{call['name']}'"
        result = tool_obj.invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    final = await llm.ainvoke(messages)
    if not isinstance(final, AIMessage) or not str(final.content).strip():
        return "✗ empty final answer"
    if TOOL_ANSWER_TOKEN not in str(final.content):
        return f"✗ final answer did not use tool result (missing '{TOOL_ANSWER_TOKEN}')"
    return "✓"


async def _run_file(
    model: BaseChatModel, prompt: str, file_info: FileInfo
) -> str:
    """File-processing payload: invoke with one attached file."""
    response = await llm_call_with_files(
        [HumanMessage(content=prompt)], [file_info], model
    )
    if response.content and str(response.content).strip():
        return "✓"
    return "✗ empty response"


async def run_model_onboarding(state: GraphState) -> dict:
    spec = ModelSpec.model_validate(state["model_spec"])

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

        # 1. Simple call
        logger.info("  simple...")
        try:
            cell_results["simple"] = await _run_simple(model, state["prompt"])
        except Exception as e:
            cell_results["simple"] = f"✗ {str(e)[:60]}"
        logger.info(f"    simple: {cell_results['simple']}")

        # 2. Tool call (full round trip)
        logger.info("  tools...")
        try:
            cell_results["tools"] = await _run_tools(model)
        except Exception as e:
            cell_results["tools"] = f"✗ {str(e)[:60]}"
        logger.info(f"    tools: {cell_results['tools']}")

        # 3. File processing — one cell per selected file
        for file_name in spec.files:
            label = f"files/{file_name}"
            logger.info(f"  {label}...")
            try:
                cell_results[label] = await _run_file(
                    model, state["prompt"], FILE_REGISTRY[file_name]
                )
            except Exception as e:
                cell_results[label] = f"✗ {str(e)[:60]}"
            logger.info(f"    {label}: {cell_results[label]}")

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
