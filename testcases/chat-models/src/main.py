import logging
from typing import Any, Callable, Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel

from uipath_langchain.chat.bedrock import UiPathChatBedrock, UiPathChatBedrockConverse
from uipath_langchain.chat.vertex import UiPathChatVertex
from uipath_langchain.chat import UiPathChatOpenAI, UiPathChat, UiPathAzureChatOpenAI

logger = logging.getLogger(__name__)


def create_test_models(max_tokens: int = 100) -> list[tuple[str, Any]]:
    """Create all test chat models with the specified max_tokens."""
    return [
        ("UiPathChatOpenAI", UiPathChatOpenAI()),
        ("UiPathChatVertex", UiPathChatVertex()),
        ("UiPathChatBedrockConverse", UiPathChatBedrockConverse()),
        ("UiPathChatBedrock", UiPathChatBedrock()),
        ("UiPathChat", UiPathChat()),
        ("UiPathAzureChatOpenAI", UiPathAzureChatOpenAI())
    ]


def ensure_model_in_results(all_model_results: dict, model_name: str) -> None:
    """Ensure a model name exists in the results dictionary."""
    if model_name not in all_model_results:
        all_model_results[model_name] = {}


def format_error_message(error: str, max_length: int = 60) -> str:
    """Format an error message, truncating if too long."""
    error_str = str(error)
    if len(error_str) > max_length:
        return f"{error_str[:max_length]}..."
    return error_str


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


class PersonInfo(BaseModel):
    """Information about a person."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    city: str = Field(description="The city where the person lives")


class TestResult:
    """Accumulates test metrics across all test runs."""
    def __init__(self):
        self.chunks = 0
        self.content_length = 0
        self.tool_calls = 0

    def add_response(self, response: Any) -> None:
        if hasattr(response, 'content') and response.content:
            self.content_length += len(response.content)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            self.tool_calls += len(response.tool_calls)

    def add_chunks(self, count: int) -> None:
        self.chunks += count

    def add_tool_calls(self, count: int) -> None:
        self.tool_calls += count


async def run_test_method(
    method: Callable,
    messages: list,
    is_async: bool,
    is_streaming: bool,
    result: TestResult,
) -> Optional[str]:
    """Run a test method and return error message if failed, None if success."""
    try:
        if is_streaming:
            if is_async:
                chunks = [c async for c in method(messages)]
            else:
                chunks = list(method(messages))
            result.add_chunks(len(chunks))
        else:
            if is_async:
                response = await method(messages)
            else:
                response = method(messages)
            result.add_response(response)
        return None
    except Exception as e:
        return str(e)


class GraphInput(BaseModel):
    """Input model for the testing graph."""
    prompt: str = Field(
        default="Count from 1 to 5.",
        description="The prompt to send to the LLM"
    )


class GraphOutput(BaseModel):
    """Output model for the testing graph."""
    success: bool
    result_summary: str
    chunks_received: Optional[int] = None
    content_length: Optional[int] = None
    tool_calls_count: Optional[int] = None


class GraphState(MessagesState):
    """State model for the testing workflow."""
    prompt: str
    success: bool
    result_summary: str
    chunks_received: Optional[int]
    content_length: Optional[int]
    tool_calls_count: Optional[int]
    model_results: dict


def prepare_input(state: GraphState) -> dict:
    """Prepare the initial state from graph input."""
    return {
        "messages": [HumanMessage(content=state["prompt"])],
        "success": True,
        "result_summary": "",
        "chunks_received": 0,
        "content_length": 0,
        "tool_calls_count": 0,
        "model_results": {},
    }


async def test_single_model_all(
    name: str,
    model: BaseChatModel,
    messages: list,
    tools: list,
    tool_messages: list,
    structured_messages: list,
) -> tuple[str, dict, TestResult]:
    """Run all tests (invoke, stream, tools, structured output) for a single model."""
    logger.info(f"\nTesting {name}...")
    model_results = {}
    result = TestResult()

    # Test invoke/ainvoke/stream/astream
    test_methods = [
        ("invoke", False, False),
        ("ainvoke", True, False),
        ("stream", False, True),
        ("astream", True, True)
    ]

    for method_name, is_async, is_streaming in test_methods:
        logger.info(f"  Testing {method_name}...")
        method = getattr(model, method_name)
        error = await run_test_method(method, messages, is_async, is_streaming, result)
        if error:
            logger.error(f"     {method_name} failed: {error}")
            model_results[method_name] = f"✗ {format_error_message(error)}"
        else:
            logger.info(f"     {method_name}: ✓")
            model_results[method_name] = "✓"

    # Test tool calling
    logger.info(f"  Testing tool_calling...")
    try:
        llm_with_tools = model.bind_tools(tools)
        chunks = []
        async for chunk in llm_with_tools.astream(tool_messages):
            chunks.append(chunk)

        accumulated = None
        for chunk in chunks:
            accumulated = chunk if accumulated is None else accumulated + chunk

        if accumulated and hasattr(accumulated, 'tool_calls') and accumulated.tool_calls:
            tool_calls_count = len(accumulated.tool_calls)
            result.add_tool_calls(tool_calls_count)
            logger.info(f"     Tool calls detected: {tool_calls_count}")
            model_results["tool_calling"] = f"✓ ({tool_calls_count} calls)"
        else:
            logger.warning(f"     No tool calls detected")
            model_results["tool_calling"] = "✗ No tool calls detected"
    except Exception as e:
        logger.error(f"     Tool calling failed: {e}")
        model_results["tool_calling"] = f"✗ {format_error_message(str(e))}"

    # Test structured output
    logger.info(f"  Testing structured_output...")
    try:
        llm_with_structure = model.with_structured_output(PersonInfo)
        response = await llm_with_structure.ainvoke(structured_messages)

        if isinstance(response, PersonInfo):
            logger.info(f"     Structured output received: {response.model_dump()}")
            model_results["structured_output"] = "✓"
        elif isinstance(response, dict):
            required_fields = {"name", "age", "city"}
            if required_fields.issubset(response.keys()):
                logger.info(f"     Structured output received (dict): {response}")
                model_results["structured_output"] = "✓"
            else:
                missing = required_fields - response.keys()
                logger.warning(f"     Dict missing required fields: {missing}")
                model_results["structured_output"] = f"✗ Missing fields: {missing}"
        else:
            logger.warning(f"     Response is not PersonInfo or dict: {type(response)}")
            model_results["structured_output"] = f"✗ Wrong type: {type(response)}"
    except Exception as e:
        logger.error(f"     Structured output failed: {e}")
        model_results["structured_output"] = f"✗ {format_error_message(str(e))}"

    return name, model_results, result


async def run_all_tests(state: GraphState) -> dict:
    """Run all tests for all chat models in parallel."""
    import asyncio

    logger.info("="*80)
    logger.info("Running All Tests")
    logger.info("="*80)

    models = create_test_models(max_tokens=2000)
    tools = [get_weather, calculate]
    tool_messages = [HumanMessage(content="What's the weather in San Francisco? Also calculate 15 * 23.")]
    structured_messages = [HumanMessage(content="Tell me about John Smith, a 35 year old software engineer living in New York.")]

    # Run all models in parallel
    tasks = [
        test_single_model_all(name, model, state["messages"], tools, tool_messages, structured_messages)
        for name, model in models
    ]
    results_list = await asyncio.gather(*tasks)

    # Aggregate results
    all_model_results = {}
    total_result = TestResult()

    for name, model_results, result in results_list:
        all_model_results[name] = model_results
        total_result.chunks += result.chunks
        total_result.content_length += result.content_length
        total_result.tool_calls += result.tool_calls

    # Build summary
    logger.info("="*80)
    summary_lines = []
    for model_name in ["UiPathChatOpenAI", "UiPathChatVertex", "UiPathChatBedrockConverse", "UiPathChatBedrock", "UiPathChat", "UiPathAzureChatOpenAI"]:
        if model_name in all_model_results:
            summary_lines.append(f"{model_name}:")
            results = all_model_results[model_name]
            for test_name in ["invoke", "ainvoke", "stream", "astream", "tool_calling", "structured_output"]:
                if test_name in results:
                    summary_lines.append(f"  {test_name}: {results[test_name]}")

    has_failures = any("✗" in str(v) for r in all_model_results.values() for v in r.values())

    return {
        "success": not has_failures,
        "result_summary": "\n".join(summary_lines),
        "chunks_received": total_result.chunks,
        "content_length": total_result.content_length,
        "tool_calls_count": total_result.tool_calls,
        "model_results": all_model_results,
    }


async def return_results(state: GraphState) -> GraphOutput:
    """Return final test results."""
    logger.info("="*80)
    logger.info("TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Success: {state['success']}")
    logger.info(f"Summary: {state['result_summary']}")
    if state.get('chunks_received'):
        logger.info(f"Chunks Received: {state['chunks_received']}")
    if state.get('content_length'):
        logger.info(f"Content Length: {state['content_length']}")
    if state.get('tool_calls_count'):
        logger.info(f"Tool Calls: {state['tool_calls_count']}")

    return GraphOutput(
        success=state["success"],
        result_summary=state["result_summary"],
        chunks_received=state.get("chunks_received"),
        content_length=state.get("content_length"),
        tool_calls_count=state.get("tool_calls_count"),
    )


def build_graph() -> StateGraph:
    """Build and compile the testing graph."""
    builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)

    builder.add_node("prepare_input", prepare_input)
    builder.add_node("run_all_tests", run_all_tests)
    builder.add_node("results", return_results)

    builder.add_edge(START, "prepare_input")
    builder.add_edge("prepare_input", "run_all_tests")
    builder.add_edge("run_all_tests", "results")
    builder.add_edge("results", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()
