import logging
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

from uipath_langchain.chat import ChatBedrockConverseUiPath, ChatGeminiUiPath, ChatOpenAIUiPath
from uipath_langchain.chat.chat_models import BedrockModels, GeminiModels, OpenAiModels

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

TestType = Literal["invoke", "streaming_with_tools", "all_apis", "streaming"]
NextNode = Literal["test_invoke", "test_streaming_with_tools", "test_all_apis", "test_streaming", "results"]


class GraphInput(BaseModel):
    """Input model for the feature testing graph."""
    test_type: TestType = Field(
        description="Type of test to run: streaming, invoke, streaming_with_tools, or all_apis"
    )
    prompt: str = Field(
        default="Count from 1 to 5.",
        description="The prompt to send to the LLM"
    )


class GraphOutput(BaseModel):
    """Output model for the feature testing graph."""
    test_type: str
    success: bool
    result_summary: str
    chunks_received: Optional[int] = None
    content_length: Optional[int] = None
    tool_calls_count: Optional[int] = None


class GraphState(MessagesState):
    """State model for the feature testing workflow."""
    test_type: TestType
    prompt: str

    # Results
    success: bool = True
    result_summary: str = ""
    chunks_received: Optional[int] = None
    content_length: Optional[int] = None
    tool_calls_count: Optional[int] = None


@tool
def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state/country, e.g. 'San Francisco, CA'
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"The weather in {location} is 72°{unit[0].upper()}"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate, e.g. '2 + 2'
    """
    try:
        result = eval(expression)
        return f"The result is {result}"
    except Exception as e:
        return f"Error calculating: {e}"


def prepare_input(graph_input: GraphInput) -> GraphState:
    """Prepare the initial state from graph input."""
    return GraphState(
        test_type=graph_input.test_type,
        prompt=graph_input.prompt,
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=graph_input.prompt)
        ],
        success=True,
        result_summary="",
    )


def route_test(state: GraphState) -> NextNode:
    """Route to the appropriate test node based on test_type."""
    test_type = state["test_type"]

    if test_type == "streaming":
        return "test_streaming"
    elif test_type == "invoke":
        return "test_invoke"
    elif test_type == "streaming_with_tools":
        return "test_streaming_with_tools"
    elif test_type == "all_apis":
        return "test_all_apis"
    else:
        return "results"


async def test_streaming(state: GraphState) -> Command:
    """Test basic streaming without tools."""
    logger.info("="*80)
    logger.info("TEST: Simple Streaming (No Tools)")
    logger.info("="*80)

    try:
        llms = [
            ChatOpenAIUiPath(
                model_name=OpenAiModels.gpt_4o_mini_2024_07_18,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            ),
            ChatGeminiUiPath(
                model_name=GeminiModels.gemini_2_0_flash_001,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            ),
            ChatBedrockConverseUiPath(
                model_name=BedrockModels.anthropic_claude_3_haiku,
                temperature=0.7,
                max_tokens=200,
            ),
        ]

        total_chunks = 0
        total_content_length = 0
        errors = []

        for llm in llms:
            logger.info(f"Testing {llm.__class__.__name__}...")
            try:
                chunks = []
                full_content = ""

                async for chunk in llm.astream(state["messages"]):
                    chunks.append(chunk)
                    content = chunk.content
                    full_content += content

                logger.info(f"{llm.__class__.__name__}: {len(chunks)} chunks, {len(full_content)} characters")
                total_chunks += len(chunks)
                total_content_length += len(full_content)
            except Exception as e:
                error_msg = f"{llm.__class__.__name__} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            error_summary = "; ".join(errors)
            logger.warning(f"Some models failed: {error_summary}")
            success_msg = f"Streaming test completed with errors. {error_summary}"
        else:
            success_msg = f"Streaming test completed successfully for all 3 models"

        logger.info(f"All streaming tests completed: {total_chunks} total chunks, {total_content_length} total characters")

        return Command(
            update={
                "success": len(errors) == 0,
                "result_summary": success_msg,
                "chunks_received": total_chunks,
                "content_length": total_content_length,
            },
            goto="results",
        )
    except Exception as e:
        logger.error(f"Streaming test failed: {str(e)}")
        return Command(
            update={
                "success": False,
                "result_summary": f"Streaming test failed: {str(e)}",
            },
            goto="results",
        )


async def test_invoke(state: GraphState) -> Command:
    """Test non-streaming mode (invoke)."""
    logger.info("="*80)
    logger.info("TEST: Non-Streaming Mode (Invoke)")
    logger.info("="*80)

    try:
        llms = [
            ChatOpenAIUiPath(
                model_name=OpenAiModels.gpt_4o_mini_2024_07_18,
                temperature=0.7,
                max_tokens=200,
                streaming=False,
            ),
            ChatGeminiUiPath(
                model_name=GeminiModels.gemini_2_0_flash_001,
                temperature=0.7,
                max_tokens=200,
                streaming=False,
            ),
            ChatBedrockConverseUiPath(
                model_name=BedrockModels.anthropic_claude_3_haiku,
                temperature=0.7,
                max_tokens=200,
            ),
        ]

        total_content_length = 0
        errors = []

        for llm in llms:
            logger.info(f"Testing {llm.__class__.__name__}...")
            try:
                response = await llm.ainvoke(state["messages"])
                content = response.content

                logger.info(f"{llm.__class__.__name__}: {len(content)} characters")
                total_content_length += len(content)
            except Exception as e:
                error_msg = f"{llm.__class__.__name__} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            error_summary = "; ".join(errors)
            logger.warning(f"Some models failed: {error_summary}")
            success_msg = f"Invoke test completed with errors. {error_summary}"
        else:
            success_msg = f"Invoke test completed successfully for all 3 models"

        logger.info(f"All invoke tests completed: {total_content_length} total characters")

        return Command(
            update={
                "success": len(errors) == 0,
                "result_summary": success_msg,
                "content_length": total_content_length,
            },
            goto="results",
        )
    except Exception as e:
        logger.error(f"Invoke test failed: {str(e)}")
        return Command(
            update={
                "success": False,
                "result_summary": f"Invoke test failed: {str(e)}",
            },
            goto="results",
        )


async def test_streaming_with_tools(state: GraphState) -> Command:
    """Test streaming with tool calling."""
    logger.info("="*80)
    logger.info("TEST: Streaming with Tool Calling")
    logger.info("="*80)

    try:
        llms = [
            ChatOpenAIUiPath(
                model_name=OpenAiModels.gpt_4o_mini_2024_07_18,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            ),
            ChatGeminiUiPath(
                model_name=GeminiModels.gemini_2_0_flash_001,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            ),
            ChatBedrockConverseUiPath(
                model_name=BedrockModels.anthropic_claude_3_haiku,
                temperature=0.7,
                max_tokens=200,
            ),
        ]

        tools = [get_weather, calculate]
        tool_messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather in San Francisco? Also calculate 15 * 23.")
        ]

        total_chunks = 0
        total_tool_calls = 0
        errors = []

        for llm in llms:
            logger.info(f"Testing {llm.__class__.__name__}...")
            try:
                llm_with_tools = llm.bind_tools(tools)

                chunks = []
                tool_call_chunk_count = 0

                async for chunk in llm_with_tools.astream(tool_messages):
                    chunks.append(chunk)
                    if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                        tool_call_chunk_count += 1

                accumulated = None
                for chunk in chunks:
                    if accumulated is None:
                        accumulated = chunk
                    else:
                        accumulated = accumulated + chunk

                tool_calls_count = 0
                if accumulated and hasattr(accumulated, 'tool_calls') and accumulated.tool_calls:
                    tool_calls_count = len(accumulated.tool_calls)
                    logger.info(f"  Tool calls detected: {tool_calls_count}")
                    for i, tc in enumerate(accumulated.tool_calls):
                        logger.info(f"    {i+1}. {tc.get('name', 'unknown')}({tc.get('args', {})})")

                logger.info(f"{llm.__class__.__name__}: {len(chunks)} chunks, {tool_call_chunk_count} tool call chunks")
                total_chunks += len(chunks)
                total_tool_calls += tool_calls_count
            except Exception as e:
                error_msg = f"{llm.__class__.__name__} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            error_summary = "; ".join(errors)
            logger.warning(f"Some models failed: {error_summary}")
            success_msg = f"Streaming with tools test completed with errors. {error_summary}"
        else:
            success_msg = f"Streaming with tools test completed successfully for all 3 models"

        logger.info(f"All streaming with tools tests completed: {total_chunks} total chunks, {total_tool_calls} total tool calls")

        return Command(
            update={
                "success": len(errors) == 0,
                "result_summary": success_msg,
                "chunks_received": total_chunks,
                "tool_calls_count": total_tool_calls,
            },
            goto="results",
        )
    except Exception as e:
        logger.error(f"Streaming with tools test failed: {str(e)}")
        return Command(
            update={
                "success": False,
                "result_summary": f"Streaming with tools test failed: {str(e)}",
            },
            goto="results",
        )


async def test_all_apis(state: GraphState) -> Command:
    """Test all 3 chat models."""
    logger.info("="*80)
    logger.info("TEST: All 3 Chat Models")
    logger.info("="*80)

    try:
        llms = [
            ("ChatOpenAIUiPath", ChatOpenAIUiPath(
                model_name=OpenAiModels.gpt_4o_mini_2024_07_18,
                temperature=0.7,
                max_tokens=100,
                streaming=True,
            )),
            ("ChatGeminiUiPath", ChatGeminiUiPath(
                model_name=GeminiModels.gemini_2_0_flash_001,
                temperature=0.7,
                max_tokens=100,
                streaming=True,
            )),
            ("ChatBedrockConverseUiPath", ChatBedrockConverseUiPath(
                model_name=BedrockModels.anthropic_claude_3_haiku,
                temperature=0.7,
                max_tokens=100,
            )),
        ]

        total_chunks = 0
        results = []
        errors = []

        for name, llm in llms:
            logger.info(f"Testing {name}...")
            try:
                chunks = []
                async for chunk in llm.astream(state["messages"]):
                    chunks.append(chunk)

                logger.info(f"  {name}: {len(chunks)} chunks")
                total_chunks += len(chunks)
                results.append(f"{name}: {len(chunks)} chunks")
            except Exception as e:
                error_msg = f"{name} failed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            error_summary = "; ".join(errors)
            logger.warning(f"Some models failed: {error_summary}")
            success_msg = f"All APIs test completed with errors. {error_summary}"
        else:
            success_msg = f"All 3 models tested successfully. {', '.join(results)}"

        logger.info(f"Testing completed for all 3 models")

        return Command(
            update={
                "success": len(errors) == 0,
                "result_summary": success_msg,
                "chunks_received": total_chunks,
            },
            goto="results",
        )
    except Exception as e:
        logger.error(f"All APIs test failed: {str(e)}")
        return Command(
            update={
                "success": False,
                "result_summary": f"All APIs test failed: {str(e)}",
            },
            goto="results",
        )


async def return_results(state: GraphState) -> GraphOutput:
    """Return final test results."""
    logger.info("="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Test Type: {state['test_type']}")
    logger.info(f"Success: {state['success']}")
    logger.info(f"Summary: {state['result_summary']}")
    if state.get('chunks_received'):
        logger.info(f"Chunks Received: {state['chunks_received']}")
    if state.get('content_length'):
        logger.info(f"Content Length: {state['content_length']}")
    if state.get('tool_calls_count'):
        logger.info(f"Tool Calls: {state['tool_calls_count']}")

    return GraphOutput(
        test_type=state["test_type"],
        success=state["success"],
        result_summary=state["result_summary"],
        chunks_received=state.get("chunks_received"),
        content_length=state.get("content_length"),
        tool_calls_count=state.get("tool_calls_count"),
    )


def build_graph() -> StateGraph:
    """Build and compile the feature testing graph."""
    builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)

    builder.add_node("prepare_input", prepare_input)
    builder.add_node("test_streaming", test_streaming)
    builder.add_node("test_invoke", test_invoke)
    builder.add_node("test_streaming_with_tools", test_streaming_with_tools)
    builder.add_node("test_all_apis", test_all_apis)
    builder.add_node("results", return_results)

    builder.add_edge(START, "prepare_input")
    builder.add_conditional_edges("prepare_input", route_test)
    builder.add_edge("results", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()
