"""Tool-calling agent demonstrating suspend/resume with RPA process invocation."""

import logging

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath.platform.common import InvokeProcess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Input(BaseModel):
    """Input for the test agent."""

    query: str


class Output(BaseModel):
    """Output from the test agent."""

    result: str


class State(BaseModel):
    """Internal state for the agent."""

    query: str
    result: str = ""


async def invoke_process_node(state: State) -> State:
    """Node that invokes an RPA process and suspends execution."""
    logger.info("=" * 80)
    logger.info("AGENT NODE: Starting invoke_process_node")
    logger.info(f"AGENT NODE: Received query: {state.query}")

    # Create an InvokeProcess request
    invoke_request = InvokeProcess(
        name="TestProcess",
        input_arguments={"query": state.query, "data": "test_data"},
        process_folder_path="Shared",
    )

    logger.info(
        f"AGENT NODE: Created InvokeProcess request: {invoke_request.model_dump()}"
    )
    logger.info("🔴 AGENT NODE: About to call interrupt() - SUSPENDING EXECUTION")
    logger.info("=" * 80)

    # Interrupt execution and capture the process output
    # The runtime will detect this and return SUSPENDED status
    # When resumed, the return value will contain the process output
    process_output = interrupt(invoke_request)

    # This code won't execute until the process completes and execution resumes
    logger.info("=" * 80)
    logger.info("🟢 AGENT NODE: Execution RESUMED after interrupt()")
    logger.info("AGENT NODE: RPA process has completed")
    logger.info(f"AGENT NODE: Process output: {process_output}")
    logger.info(f"AGENT NODE: Returning result for query: {state.query}")
    logger.info("=" * 80)

    # Extract result from process output
    result = (
        process_output.get("result", "Process completed")
        if isinstance(process_output, dict)
        else str(process_output)
    )
    return State(query=state.query, result=result)


# Build the graph
builder = StateGraph(state_schema=State)

# Add single node that invokes the process
builder.add_node("invoke_process", invoke_process_node)

# Connect: START -> invoke_process -> END
builder.add_edge(START, "invoke_process")
builder.add_edge("invoke_process", END)

# Compile with checkpointer (required for interrupts to work)
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
