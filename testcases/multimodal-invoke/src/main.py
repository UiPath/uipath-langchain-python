import logging
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from pydantic import BaseModel, Field
from uipath.llm_client.settings import PlatformSettings
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
)
from uipath_langchain_client.settings import (
    ApiFlavor,
    RoutingMode,
    UiPathBaseSettings,
    VendorType,
)

from uipath_langchain.agent.multimodal.invoke import llm_call_with_files
from uipath_langchain.agent.multimodal.types import FileInfo
from uipath_langchain.chat.chat_model_factory import get_chat_model

logger = logging.getLogger(__name__)


@dataclass
class ModelTestConfig:
    """Configuration for a model test case targeting a specific factory class."""

    label: str
    model_name: str
    vendor_type: VendorType | None = None
    api_flavor: ApiFlavor | None = None
    routing_mode: RoutingMode = RoutingMode.PASSTHROUGH
    model_factory: Callable[..., object] | None = field(default=None)


def _create_model(config: ModelTestConfig, client_settings: UiPathBaseSettings) -> object:
    """Create a model instance, using custom factory if provided."""
    if config.model_factory is not None:
        return config.model_factory(
            model_name=config.model_name,
            settings=client_settings,
            temperature=0.0,
            max_tokens=200,
        )
    return get_chat_model(
        model=config.model_name,
        client_settings=client_settings,
        routing_mode=config.routing_mode,
        vendor_type=config.vendor_type,
        api_flavor=config.api_flavor,
        temperature=0.0,
        max_tokens=200,
    )


# Each entry targets a different class returned by get_chat_model_factory
MODELS_TO_TEST: list[ModelTestConfig] = [
    # VendorType.OPENAI (UiPath-owned) -> UiPathAzureChatOpenAI
    ModelTestConfig(
        label="OpenAI (Azure) - UiPathAzureChatOpenAI",
        model_name="gpt-5.2-2025-12-11",
    ),
    # VendorType.OPENAI (UiPath-owned) + RESPONSES -> UiPathAzureChatOpenAI (responses API)
    ModelTestConfig(
        label="OpenAI (Azure, Responses API) - UiPathAzureChatOpenAI",
        model_name="gpt-5.2-2025-12-11",
        api_flavor=ApiFlavor.RESPONSES,
    ),
    # VendorType.VERTEXAI (Google family) -> UiPathChatGoogleGenerativeAI
    ModelTestConfig(
        label="VertexAI (Google) - UiPathChatGoogleGenerativeAI",
        model_name="gemini-2.5-pro",
    ),
    # VendorType.VERTEXAI (Google family) -> UiPathChatGoogleGenerativeAI
    ModelTestConfig(
        label="VertexAI (Google) - UiPathChatGoogleGenerativeAI",
        model_name="gemini-3-pro-preview",
    ),
    # Direct instantiation -> UiPathChatAnthropicBedrock
    ModelTestConfig(
        label="Bedrock (Anthropic, direct) - UiPathChatAnthropicBedrock",
        model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    # Direct instantiation -> UiPathChatBedrockConverse
    ModelTestConfig(
        label="Bedrock (Converse, direct) - UiPathChatBedrockConverse",
        model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
        api_flavor=ApiFlavor.CONVERSE,
    ),
    # Direct instantiation -> UiPathChatBedrock
    ModelTestConfig(
        label="Bedrock (Invoke, direct) - UiPathChatBedrock",
        model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
        api_flavor=ApiFlavor.INVOKE,
    ),
]

FILES_TO_TEST = [
    FileInfo(
        url="https://www.w3schools.com/css/img_5terre.jpg",
        name="img_5terre.jpg",
        mime_type="image/jpeg",
    ),
    FileInfo(
        url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        name="dummy.pdf",
        mime_type="application/pdf",
    ),
]


class GraphInput(BaseModel):
    prompt: str = Field(default="Describe the content of this file in one sentence.")


class GraphOutput(BaseModel):
    success: bool
    result_summary: str


class GraphState(MessagesState):
    prompt: str
    success: bool
    result_summary: str
    model_results: dict


async def run_multimodal_invoke(state: GraphState) -> dict:
    messages = [HumanMessage(content=state["prompt"])]
    model_results = {}

    for config in MODELS_TO_TEST:
        logger.info(f"Testing {config.label}...")

        client_settings = PlatformSettings(agenthub_config="agentsplayground")

        model = _create_model(config, client_settings)
        logger.info(f"  Created: {type(model).__name__}")
        test_results = {}
        for file_info in FILES_TO_TEST:
            label = file_info.name
            logger.info(f"  {label}...")
            try:
                response: AIMessage = await llm_call_with_files(
                    messages, [file_info], model
                )
                logger.info(f"    {label}: ✓")
                test_results[label] = "✓"
            except Exception as e:
                logger.error(f"    {label}: ✗ {e}")
                test_results[label] = f"✗ {str(e)[:60]}"
        model_results[config.label] = test_results

    summary_lines = []
    for label, results in model_results.items():
        summary_lines.append(f"{label}:")
        for file_name, result in results.items():
            summary_lines.append(f"  {file_name}: {result}")
    has_failures = any(
        "✗" in v for results in model_results.values() for v in results.values()
    )

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

    builder.add_node("run_multimodal_invoke", run_multimodal_invoke)
    builder.add_node("results", return_results)

    builder.add_edge(START, "run_multimodal_invoke")
    builder.add_edge("run_multimodal_invoke", "results")
    builder.add_edge("results", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()
