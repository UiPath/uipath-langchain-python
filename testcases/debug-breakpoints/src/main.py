"""Debug Breakpoints Test Agent - Tests static breakpoint functionality via uipath debug."""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GraphInput(BaseModel):
    """Input model for the debug breakpoints test graph."""
    task: str = Field(description="The task to process")
    value: int = Field(description="A numeric value to process")


class GraphOutput(BaseModel):
    """Output model for the debug breakpoints test graph."""
    task: str = Field(description="The original task")
    initial_value: int = Field(description="The initial input value")
    processed_value: int = Field(description="The final processed value")
    steps_completed: list[str] = Field(description="List of completed processing steps")


class GraphState(BaseModel):
    """State model for the debug breakpoints workflow."""
    task: str
    initial_value: int
    current_value: int
    steps_completed: list[str] = Field(default_factory=list)


def prepare_input(graph_input: GraphInput) -> GraphState:
    """Prepare the initial state from graph input."""
    logger.info(f"Preparing input: task={graph_input.task}, value={graph_input.value}")
    return GraphState(
        task=graph_input.task,
        initial_value=graph_input.value,
        current_value=graph_input.value,
        steps_completed=[],
    )


def process_step_1(state: GraphState) -> Command:
    """First processing step - doubles the value."""
    logger.info(f"Step 1: Processing value {state.current_value}")

    new_value = state.current_value * 2
    steps = state.steps_completed + ["step_1_double"]

    logger.info(f"Step 1 complete: {state.current_value} -> {new_value}")

    return Command(
        update={
            "current_value": new_value,
            "steps_completed": steps,
        }
    )


def process_step_2(state: GraphState) -> Command:
    """Second processing step - adds 100 to the value."""
    logger.info(f"Step 2: Processing value {state.current_value}")

    new_value = state.current_value + 100
    steps = state.steps_completed + ["step_2_add_100"]

    logger.info(f"Step 2 complete: {state.current_value} -> {new_value}")

    return Command(
        update={
            "current_value": new_value,
            "steps_completed": steps,
        }
    )


def process_step_3(state: GraphState) -> Command:
    """Third processing step - multiplies by 3."""
    logger.info(f"Step 3: Processing value {state.current_value}")

    new_value = state.current_value * 3
    steps = state.steps_completed + ["step_3_multiply_3"]

    logger.info(f"Step 3 complete: {state.current_value} -> {new_value}")

    return Command(
        update={
            "current_value": new_value,
            "steps_completed": steps,
        }
    )


def process_step_4(state: GraphState) -> Command:
    """Fourth processing step - subtracts 50."""
    logger.info(f"Step 4: Processing value {state.current_value}")

    new_value = state.current_value - 50
    steps = state.steps_completed + ["step_4_subtract_50"]

    logger.info(f"Step 4 complete: {state.current_value} -> {new_value}")

    return Command(
        update={
            "current_value": new_value,
            "steps_completed": steps,
        }
    )


def process_step_5(state: GraphState) -> Command:
    """Fifth processing step - adds 10."""
    logger.info(f"Step 5: Processing value {state.current_value}")

    new_value = state.current_value + 10
    steps = state.steps_completed + ["step_5_add_10"]

    logger.info(f"Step 5 complete: {state.current_value} -> {new_value}")

    return Command(
        update={
            "current_value": new_value,
            "steps_completed": steps,
        }
    )


def finalize(state: GraphState) -> GraphOutput:
    """Finalize the workflow and return output."""
    logger.info(f"Finalizing: task={state.task}, final_value={state.current_value}")

    return GraphOutput(
        task=state.task,
        initial_value=state.initial_value,
        processed_value=state.current_value,
        steps_completed=state.steps_completed,
    )


def build_graph() -> StateGraph:
    """Build and compile the debug breakpoints test graph.

    Graph flow:
    START -> prepare_input -> step_1 -> step_2 -> step_3 -> step_4 -> step_5 -> finalize -> END

    This graph is designed to test static breakpoints set via `uipath debug`:
    - Set breakpoint: `b process_step_2`
    - Step through: `s` (step mode)
    - Continue: `c`
    - List breakpoints: `l`
    - Remove breakpoint: `r process_step_2`
    - Quit: `q`

    With input value=10:
    - After step_1: 10 * 2 = 20
    - After step_2: 20 + 100 = 120
    - After step_3: 120 * 3 = 360
    - After step_4: 360 - 50 = 310
    - After step_5: 310 + 10 = 320
    """
    builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)

    # Add nodes
    builder.add_node("prepare_input", prepare_input)
    builder.add_node("process_step_1", process_step_1)
    builder.add_node("process_step_2", process_step_2)
    builder.add_node("process_step_3", process_step_3)
    builder.add_node("process_step_4", process_step_4)
    builder.add_node("process_step_5", process_step_5)
    builder.add_node("finalize", finalize)

    # Add edges - simple linear flow
    builder.add_edge(START, "prepare_input")
    builder.add_edge("prepare_input", "process_step_1")
    builder.add_edge("process_step_1", "process_step_2")
    builder.add_edge("process_step_2", "process_step_3")
    builder.add_edge("process_step_3", "process_step_4")
    builder.add_edge("process_step_4", "process_step_5")
    builder.add_edge("process_step_5", "finalize")
    builder.add_edge("finalize", END)

    # Compile with memory checkpointer for state persistence
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Create the compiled graph
graph = build_graph()
