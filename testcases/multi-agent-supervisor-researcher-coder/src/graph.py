import os
from typing import Annotated, Literal, Union

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from uipath_langchain.chat import UiPathAzureChatOpenAI, UiPathChat

# Configuration constants
MAX_SEARCH_RESULTS = 5
DEFAULT_MODEL = "gpt-4o-2024-08-06"
ALTERNATIVE_MODEL = "claude-3-5-sonnet-latest"
RECURSION_LIMIT = 50

# Team configuration
TEAM_MEMBERS = ["researcher", "coder"]
ROUTING_OPTIONS = TEAM_MEMBERS + ["FINISH"]


def get_search_tool() -> Union[TavilySearchResults, DuckDuckGoSearchResults]:
    """Get the appropriate search tool based on available API keys."""
    if os.getenv("TAVILY_API_KEY"):
        return TavilySearchResults(max_results=MAX_SEARCH_RESULTS)
    return DuckDuckGoSearchResults()


def create_python_repl() -> PythonREPL:
    """Create Python REPL instance with safety warning."""
    # WARNING: This executes code locally, which can be unsafe when not sandboxed
    return PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """
    repl = create_python_repl()
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str


def create_supervisor_prompt() -> str:
    """Create the system prompt for the supervisor agent."""
    return (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {TEAM_MEMBERS}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
        "\n\nRules:"
        "\n- If a worker has completed their task and provided useful results, consider if another worker is needed"
        "\n- If the user's question has been fully answered with research and/or code, respond with FINISH"
        "\n- Don't route to the same worker repeatedly if they've already completed their task"
    )


def create_llm() -> Union[UiPathAzureChatOpenAI, UiPathChat]:
    """Create and configure the language model based on an environment variable."""
    if os.getenv("USE_AZURE_CHAT", "false").lower() == "true":
        return UiPathAzureChatOpenAI(model=DEFAULT_MODEL)
    return UiPathChat(model=DEFAULT_MODEL)


# Type definitions
SupervisorOptions = Literal["researcher", "coder", "FINISH"]
TeamMembers = Literal["researcher", "coder"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: SupervisorOptions


class GraphInput(BaseModel):
    """Input model for the multi-agent graph."""
    question: str


class GraphOutput(BaseModel):
    """Output model for the multi-agent graph."""
    answer: str


class State(MessagesState):
    """Extended state with routing information."""
    next: str
    workers_called: list[str] = []
    


def create_input_state(state: GraphInput) -> dict:
    """Create initial state with system prompt and user question."""
    system_prompt = create_supervisor_prompt()
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.question),
        ],
        "next": "",
        "workers_called": [],
    }

async def supervisor_node(state: State) -> Command[TeamMembers] | GraphOutput:
    """Supervisor node that routes between team members or finishes the task."""
    # Add context about which workers have been called
    context_messages = state["messages"].copy()
    if state.get("workers_called"):
        context_msg = f"\n\nWorkers already called: {', '.join(state['workers_called'])}"
        context_messages[-1].content += context_msg
    
    llm = create_llm()
    response = await llm.with_structured_output(Router).ainvoke(context_messages)
    goto = response["next"]
    print("Routing to:", goto)
    
    if goto == "FINISH":
        # Find the last meaningful response from a worker
        last_worker_message = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'name') and msg.name in ["researcher", "coder"]:
                last_worker_message = msg.content
                break
        
        answer = last_worker_message if last_worker_message else state["messages"][-1].content
        return GraphOutput(answer=answer)
    else:
        # Update workers_called list
        current_workers = state.get("workers_called", [])
        if goto not in current_workers:
            current_workers = current_workers + [goto]
        
        return Command(goto=goto, update={"next": goto, "workers_called": current_workers})


def create_research_agent():
    """Create the research agent with search capabilities."""
    llm = create_llm()
    search_tool = get_search_tool()
    agent = create_react_agent(
        llm, 
        tools=[search_tool], 
        prompt="""You are a researcher. Your job is to gather information and provide comprehensive answers to questions.

Rules:
- Use the search tool to find relevant information
- Provide detailed, helpful responses based on your research
- DO NOT perform mathematical calculations - that's the coder's job
- Once you've gathered sufficient information to answer the question, provide your findings
- Be thorough but concise in your responses"""
    )
    return agent.with_config({"recursion_limit": RECURSION_LIMIT})


async def research_node(state: State) -> Command[Literal["supervisor"]]:
    """Research node that performs information gathering."""
    research_agent = create_research_agent()
    result = await research_agent.ainvoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )


def create_code_agent():
    """Create the code agent with Python execution capabilities.
    
    WARNING: This performs arbitrary code execution, which can be unsafe when not sandboxed.
    """
    llm = create_llm()
    agent = create_react_agent(
        llm, 
        tools=[python_repl_tool],
        prompt="""You are a coder. Your job is to write and execute Python code to solve problems, perform calculations, and generate visualizations.

Rules:
- Use the python_repl_tool to execute code
- Write clear, well-commented code
- Print results so they're visible
- If you need data that should be researched first, let the supervisor know
- Provide explanations of your code and results"""
    )
    return agent.with_config({"recursion_limit": RECURSION_LIMIT})


async def code_node(state: State) -> Command[Literal["supervisor"]]:
    """Code node that executes Python code and performs calculations."""
    code_agent = create_code_agent()
    result = await code_agent.ainvoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )


def build_multi_agent_graph() -> StateGraph:
    """Build and compile the multi-agent supervisor graph."""
    builder = StateGraph(State, input=GraphInput, output=GraphOutput)
    
    # Add nodes
    builder.add_node("input", create_input_state)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("coder", code_node)
    
    # Add edges
    builder.add_edge(START, "input")
    builder.add_edge("input", "supervisor")
    
    return builder.compile().with_config({"recursion_limit": RECURSION_LIMIT})


# Create the compiled graph
graph = build_multi_agent_graph()
