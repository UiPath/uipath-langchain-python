import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from pydantic import BaseModel, Field

from uipath_langchain.agent.multimodal.invoke import llm_call_with_files
from uipath_langchain.agent.multimodal.types import FileInfo
from uipath_langchain.chat.chat_model_factory import get_chat_model

logger = logging.getLogger(__name__)

MODELS_TO_TEST = [
    "gpt-4.1-2025-04-14",
    "gemini-2.5-pro",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
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

    for model_name in MODELS_TO_TEST:
        logger.info(f"Testing {model_name}...")
        model = get_chat_model(
            model=model_name,
            temperature=0.0,
            max_tokens=200,
            agenthub_config="agentsplayground",
        )
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
        model_results[model_name] = test_results

    summary_lines = []
    for model_name, results in model_results.items():
        summary_lines.append(f"{model_name}:")
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
