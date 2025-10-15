import random
from enum import Enum
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic.dataclasses import dataclass
from uipath.tracing import traced
from uipath.eval.mocks import mockable, ExampleCall


class Operator(Enum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    RANDOM = "random"


@dataclass
class CalculatorInput:
    a: float
    b: float
    operator: Operator


@dataclass
class CalculatorOutput:
    result: float

GET_RANDOM_OPERATOR_EXAMPLES = [ExampleCall(id="example", input="{}", output="{\"result\": \"*\"}")]

@traced()
@mockable(example_calls=GET_RANDOM_OPERATOR_EXAMPLES)
async def get_random_operator() -> Operator:
    """Return a random math operator."""
    return random.choice([
        Operator.ADD,
        Operator.SUBTRACT,
        Operator.MULTIPLY,
        Operator.DIVIDE,
    ])


@traced()
async def postprocess(x: float) -> float:
    """Example of nested traced invocation."""
    return x


@traced()
async def calculate(input: CalculatorInput) -> CalculatorOutput:
    if input.operator == Operator.RANDOM:
        operator = await get_random_operator()
    else:
        operator = input.operator

    match operator:
        case Operator.ADD:
            result = input.a + input.b
        case Operator.SUBTRACT:
            result = input.a - input.b
        case Operator.MULTIPLY:
            result = input.a * input.b
        case Operator.DIVIDE:
            result = input.a / input.b if input.b != 0 else 0
        case _:
            raise ValueError(f"Unknown operator: {operator}")

    result = await postprocess(result)
    return CalculatorOutput(result=result)


builder = StateGraph(state_schema=CalculatorInput, input=CalculatorInput, output=CalculatorOutput)

builder.add_node("calculate", calculate)
builder.add_edge(START, "calculate")
builder.add_edge("calculate", END)

graph = builder.compile()
