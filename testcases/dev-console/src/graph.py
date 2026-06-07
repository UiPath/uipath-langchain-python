"""Calculator Agent for uipath dev TUI testing."""

from enum import Enum

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic.dataclasses import dataclass
from uipath.tracing import traced


class Operator(Enum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"


@dataclass
class CalculatorInput:
    a: float
    b: float
    operator: Operator


@dataclass
class CalculatorOutput:
    result: float


@traced(name="postprocess")
async def postprocess(x: float) -> float:
    """Example of nested traced invocation."""
    return x


@traced(name="calculate")
async def calculate(input: CalculatorInput) -> CalculatorOutput:
    result = 0
    match input.operator:
        case Operator.ADD:
            result = input.a + input.b
        case Operator.SUBTRACT:
            result = input.a - input.b
        case Operator.MULTIPLY:
            result = input.a * input.b
        case Operator.DIVIDE:
            result = input.a / input.b if input.b != 0 else 0
    result = await postprocess(result)
    return CalculatorOutput(result=result)


builder = StateGraph(
    state_schema=CalculatorInput, input=CalculatorInput, output=CalculatorOutput
)

builder.add_node("calculate", calculate)
builder.add_edge(START, "calculate")
builder.add_edge("calculate", END)

graph = builder.compile()
