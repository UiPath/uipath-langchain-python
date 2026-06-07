from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.language_models.chat_models import BaseChatModel
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
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status.\n\n"
    "When to choose FINISH:\n"
    "- If the user's question has been fully answered\n"
    "- If a worker has provided a complete solution\n"
    "- If no additional work is needed\n\n"
    "When to choose a worker:\n"
    "- researcher: For searching information, finding facts, or research tasks\n"
    "- coder: For mathematical calculations, data analysis, or code execution\n\n"
    "Avoid sending workers back and forth unnecessarily. Once a worker completes the task, choose FINISH."
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
    answer: str

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
        "answer": "",
    }

def make_supervisor_node(model: BaseChatModel):
    # Wrapper to identify the node as model-based
    supervisor_llm = model.with_structured_output(Router)
    async def supervisor_node(state: State) -> dict:
        response = await supervisor_llm.ainvoke(state["messages"])
        goto = response["next"]

        # When finishing, extract the answer and store it in state
        if goto == "FINISH":
            # Get the last message from a worker (not system message)
            last_worker_message = None
            for msg in reversed(state["messages"]):
                if msg.type == "human" and hasattr(msg, "name") and msg.name in members:
                    last_worker_message = msg
                    break

            if last_worker_message:
                answer = get_message_text(last_worker_message)
            else:
                # Fallback: get last non-system message
                answer = get_message_text(state["messages"][-1])

            return {"next": goto, "answer": answer}
        else:
            return {"next": goto}

    return supervisor_node

def route_supervisor(state: State) -> Literal["researcher", "coder", "__end__"]:
    next_node = state.get("next", "")
    if next_node == "researcher":
        return "researcher"
    elif next_node == "coder":
        return "coder"
    elif next_node == "FINISH":
        return "__end__"
    else:
        return "__end__"

def output_node(state: State) -> GraphOutput:
    return GraphOutput(answer=state.get("answer", ""))

research_agent = create_agent(
    llm,
    tools=[tavily_tool],
    system_prompt=(
        "You are a researcher. DO NOT do any math. "
        "Search for information and provide findings. "
        "When you've completed your research, clearly state your findings."
    )
)


async def research_node(state: State):
    result = await research_agent.ainvoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="researcher")
        ]
    }


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
code_agent = create_agent(
    llm,
    tools=[python_repl_tool],
    system_prompt=(
        "You are a coder. Execute Python code to solve problems. "
        "When you've successfully completed the calculation or task, "
        "provide the final answer clearly."
    )
)


async def code_node(state: State):
    result = await code_agent.ainvoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="coder")
        ]
    }


builder = StateGraph(State, input=GraphInput, output=GraphOutput)
builder.add_node("input", input)
builder.add_node("supervisor", make_supervisor_node(llm))
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
builder.add_node("output", output_node)

builder.add_edge(START, "input")
builder.add_edge("input", "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor, {
    "researcher": "researcher",
    "coder": "coder",
    "__end__": "output"
})
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")
builder.add_edge("output", END)

graph = builder.compile()
