"""Test agent for suspend/resume with RPA process invocation."""

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel


class InvokeProcess(BaseModel):
    """Model representing an RPA process invocation request."""

    name: str
    input_arguments: dict
    process_folder_path: str = "Shared"
    process_folder_key: str | None = None


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
    # Create an InvokeProcess request
    invoke_request = InvokeProcess(
        name="TestProcess",
        input_arguments={"query": state.query, "data": "test_data"},
        process_folder_path="Shared",
    )

    # Interrupt execution - this will suspend the agent
    # The runtime will detect this and return SUSPENDED status
    interrupt(invoke_request)

    # This code won't execute until the process completes and execution resumes
    # For now, just return the state as-is
    return State(query=state.query, result="Process invoked, awaiting completion")


# Build the graph
builder = StateGraph(state_schema=State, input=Input, output=Output)

# Add single node that invokes the process
builder.add_node("invoke_process", invoke_process_node)

# Connect: START -> invoke_process -> END
builder.add_edge(START, "invoke_process")
builder.add_edge("invoke_process", END)

# Compile with checkpointer (required for interrupts to work)
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
