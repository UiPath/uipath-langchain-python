"""Basic agent without suspend/resume for testing eval run updates."""

import logging

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Input(BaseModel):
    """Input for the basic agent."""

    query: str


class Output(BaseModel):
    """Output from the basic agent."""

    result: str


class State(BaseModel):
    """Internal state for the agent."""

    query: str
    result: str = ""


async def process_node(state: State) -> State:
    """Node that processes the query without suspension."""
    logger.info("=" * 80)
    logger.info("BASIC AGENT: Starting process_node")
    logger.info(f"BASIC AGENT: Received query: {state.query}")

    # Simple processing - just echo back with a prefix
    result = f"Processed: {state.query}"

    logger.info(f"BASIC AGENT: Returning result: {result}")
    logger.info("=" * 80)

    return State(query=state.query, result=result)


# Build the graph
builder = StateGraph(state_schema=State)

# Add single node that processes
builder.add_node("process", process_node)

# Connect: START -> process -> END
builder.add_edge(START, "process")
builder.add_edge("process", END)

# Compile with checkpointer for consistency with other agents
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def _create_graph():
    checkpointer = AsyncSqliteSaver.from_conn_string("__uipath/state.db")
    return builder.compile(checkpointer=checkpointer)


# For synchronous access, use MemorySaver as fallback
from langgraph.checkpoint.memory import MemorySaver

graph_basic = builder.compile(checkpointer=MemorySaver())
