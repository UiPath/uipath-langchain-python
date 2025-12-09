from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain.agents import create_agent
from pydantic import BaseModel
from typing_extensions import TypedDict


tavily_tool = TavilySearch(max_results=5)

# This executes code locally, which can be unsafe
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

members = ["researcher", "coder"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


llm = ChatAnthropic(model="claude-3-7-sonnet-latest")

class GraphInput(BaseModel):
    question: str

class GraphOutput(BaseModel):
    answer: str

class State(MessagesState):
    next: str

def get_message_text(msg: BaseMessage) -> str:
    """LangChain-style safe message text extractor."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return "".join(
            block.get("text", "") for block in msg.content if block.get("type") == "text"
        )
    return ""

def input(state: GraphInput):
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.question),
        ],
        "next": "",
    }

async def supervisor_node(state: State) -> dict | GraphOutput:
    response = await llm.with_structured_output(Router).ainvoke(state["messages"])
    goto = response["next"]
    if goto == "FINISH":
        return GraphOutput(answer=get_message_text(state["messages"][-1]))
    else:
        return {"next": goto}

def route_supervisor(state: State) -> Literal["researcher", "coder"] | Literal["__end__"]:
    next_node = state.get("next", "")
    if next_node == "researcher":
        return "researcher"
    elif next_node == "coder":
        return "coder"
    else:
        return END

research_agent = create_agent(
    llm, tools=[tavily_tool], system_prompt="You are a researcher. DO NOT do any math."
)


async def research_node(state: State):
    result = await research_agent.ainvoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="researcher")
        ]
    }


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
code_agent = create_agent(llm, tools=[python_repl_tool])


async def code_node(state: State):
    result = await code_agent.ainvoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="coder")
        ]
    }


builder = StateGraph(State, input=GraphInput, output=GraphOutput)
builder.add_node("input", input)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)

builder.add_edge(START, "input")
builder.add_edge("input", "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "coder": "coder",
    END: END
})
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()
