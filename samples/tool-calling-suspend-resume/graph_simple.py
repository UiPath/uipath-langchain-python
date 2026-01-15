"""Simple suspend/resume demonstration without requiring RPA process.

This version uses a simple dict payload for interrupt() instead of InvokeProcess,
making it easier to test without requiring UiPath authentication or a real process.
"""

import logging

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

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


async def suspend_node(state: State) -> State:
    """Node that suspends execution using a simple dict payload."""
    logger.info("=" * 80)
    logger.info("AGENT NODE: Starting suspend_node")
    logger.info(f"AGENT NODE: Received query: {state.query}")
    logger.info("🔴 AGENT NODE: About to call interrupt() - SUSPENDING EXECUTION")
    logger.info("=" * 80)

    # Interrupt with simple dict (no RPA invocation needed)
    resume_data = interrupt({"message": "Waiting for external completion", "query": state.query})

    # This code won't execute until someone resumes with data
    logger.info("=" * 80)
    logger.info("🟢 AGENT NODE: Execution RESUMED after interrupt()")
    logger.info(f"AGENT NODE: Received resume data: {resume_data}")
    logger.info("=" * 80)

    # Use the resume data in the result
    result = f"Completed with resume data: {resume_data}"
    return State(query=state.query, result=result)


# Build the graph
builder = StateGraph(state_schema=State)

# Add single node that suspends
builder.add_node("suspend_node", suspend_node)

# Connect: START -> suspend_node -> END
builder.add_edge(START, "suspend_node")
builder.add_edge("suspend_node", END)

# Compile with SQLite checkpointer (required for interrupts and persistence)
# State will be saved to __uipath/state.db for resume across process restarts
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def _create_graph():
    checkpointer = AsyncSqliteSaver.from_conn_string("__uipath/state.db")
    return builder.compile(checkpointer=checkpointer)

# For synchronous access, use MemorySaver as fallback
# (Runtime will replace with proper AsyncSqliteSaver)
from langgraph.checkpoint.memory import MemorySaver
graph_simple = builder.compile(checkpointer=MemorySaver())
