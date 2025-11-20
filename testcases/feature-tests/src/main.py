"""Feature Testing Agent for UiPath LangChain capabilities.

This agent tests various features including:
- LLM streaming capabilities
- LLM non-streaming (invoke) capabilities
- Tool calling with streaming
- Both UiPathChat (normalized) and UiPathAzureChatOpenAI (passthrough) APIs
"""

import logging
import os
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.types import Command
from pydantic import BaseModel, Field

from uipath_langchain.chat import UiPathAzureChatOpenAI, UiPathChat

# Configuration
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

# Test types
TestType = Literal["streaming", "invoke", "streaming_with_tools", "both_apis"]
NextNode = Literal["test_streaming", "test_invoke", "test_streaming_with_tools", "test_both_apis", "results"]


# Data Models
class GraphInput(BaseModel):
    """Input model for the feature testing graph."""
    test_type: TestType = Field(
        description="Type of test to run: streaming, invoke, streaming_with_tools, or both_apis"
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


# Test Tools
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


# Node Functions
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
    elif test_type == "both_apis":
        return "test_both_apis"
    else:
        return "results"


async def test_streaming(state: GraphState) -> Command:
    """Test basic streaming without tools."""
    logger.info("="*80)
    logger.info("TEST: Simple Streaming (No Tools)")
    logger.info("="*80)

    try:
        # Create LLM based on environment variable
        if os.getenv("USE_AZURE_CHAT", "false").lower() == "true":
            llm = UiPathAzureChatOpenAI(
                model=DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            )
        else:
            llm = UiPathChat(
                model=DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            )

        # Stream the response
        chunks = []
        full_content = ""

        async for chunk in llm.astream(state["messages"]):
            chunks.append(chunk)
            content = chunk.content
            full_content += content

        logger.info(f"Streaming completed: {len(chunks)} chunks, {len(full_content)} characters")

        return Command(
            update={
                "success": True,
                "result_summary": f"Streaming test completed successfully",
                "chunks_received": len(chunks),
                "content_length": len(full_content),
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
        # Create LLM based on environment variable (disable streaming)
        if os.getenv("USE_AZURE_CHAT", "false").lower() == "true":
            llm = UiPathAzureChatOpenAI(
                model=DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=200,
                streaming=False,
            )
        else:
            llm = UiPathChat(
                model=DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=200,
                streaming=False,
            )

        # Invoke the LLM
        response = await llm.ainvoke(state["messages"])
        content = response.content

        logger.info(f"Invoke completed: {len(content)} characters")

        return Command(
            update={
                "success": True,
                "result_summary": f"Invoke test completed successfully",
                "content_length": len(content),
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
        # Create LLM based on environment variable
        if os.getenv("USE_AZURE_CHAT", "false").lower() == "true":
            llm = UiPathAzureChatOpenAI(
                model=DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            )
        else:
            llm = UiPathChat(
                model=DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=200,
                streaming=True,
            )

        # Bind tools
        tools = [get_weather, calculate]
        llm_with_tools = llm.bind_tools(tools)

        # Use a prompt that triggers tool calling
        tool_messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather in San Francisco? Also calculate 15 * 23.")
        ]

        # Stream the response
        chunks = []
        tool_call_chunk_count = 0

        async for chunk in llm_with_tools.astream(tool_messages):
            chunks.append(chunk)
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                tool_call_chunk_count += 1

        # Accumulate all chunks to see final tool calls
        accumulated = None
        for chunk in chunks:
            if accumulated is None:
                accumulated = chunk
            else:
                accumulated = accumulated + chunk

        tool_calls_count = 0
        if accumulated and hasattr(accumulated, 'tool_calls') and accumulated.tool_calls:
            tool_calls_count = len(accumulated.tool_calls)
            logger.info(f"Tool calls detected: {tool_calls_count}")
            for i, tc in enumerate(accumulated.tool_calls):
                logger.info(f"  {i+1}. {tc.get('name', 'unknown')}({tc.get('args', {})})")

        logger.info(f"Streaming with tools completed: {len(chunks)} chunks, {tool_call_chunk_count} tool call chunks")

        return Command(
            update={
                "success": True,
                "result_summary": f"Streaming with tools test completed successfully",
                "chunks_received": len(chunks),
                "tool_calls_count": tool_calls_count,
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


async def test_both_apis(state: GraphState) -> Command:
    """Test both UiPathChat (normalized) and UiPathAzureChatOpenAI (passthrough) APIs."""
    logger.info("="*80)
    logger.info("TEST: Compare Normalized vs Passthrough APIs")
    logger.info("="*80)

    try:
        # Test UiPathChat (normalized API)
        logger.info("Testing UiPathChat (Normalized API)")
        llm_normalized = UiPathChat(
            model=DEFAULT_MODEL,
            temperature=0.7,
            max_tokens=100,
            streaming=True,
        )

        chunks_normalized = []
        async for chunk in llm_normalized.astream(state["messages"]):
            chunks_normalized.append(chunk)

        # Test UiPathAzureChatOpenAI (passthrough API)
        logger.info("Testing UiPathAzureChatOpenAI (Passthrough API)")
        llm_passthrough = UiPathAzureChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=0.7,
            max_tokens=100,
            streaming=True,
        )

        chunks_passthrough = []
        async for chunk in llm_passthrough.astream(state["messages"]):
            chunks_passthrough.append(chunk)

        logger.info(f"Both APIs tested successfully")
        logger.info(f"  Normalized API: {len(chunks_normalized)} chunks")
        logger.info(f"  Passthrough API: {len(chunks_passthrough)} chunks")

        return Command(
            update={
                "success": True,
                "result_summary": (
                    f"Both APIs tested successfully. "
                    f"Normalized: {len(chunks_normalized)} chunks, "
                    f"Passthrough: {len(chunks_passthrough)} chunks"
                ),
                "chunks_received": len(chunks_normalized) + len(chunks_passthrough),
            },
            goto="results",
        )
    except Exception as e:
        logger.error(f"Both APIs test failed: {str(e)}")
        return Command(
            update={
                "success": False,
                "result_summary": f"Both APIs test failed: {str(e)}",
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

    # Add nodes
    builder.add_node("prepare_input", prepare_input)
    builder.add_node("test_streaming", test_streaming)
    builder.add_node("test_invoke", test_invoke)
    builder.add_node("test_streaming_with_tools", test_streaming_with_tools)
    builder.add_node("test_both_apis", test_both_apis)
    builder.add_node("results", return_results)

    # Add edges
    builder.add_edge(START, "prepare_input")
    builder.add_conditional_edges("prepare_input", route_test)
    builder.add_edge("results", END)

    # Compile with memory checkpointer
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Create the compiled graph
graph = build_graph()
