"""Parent LangGraph workflow with an embedded UiPath DeepAgent node."""

from typing import Annotated

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from uipath_langchain.chat import UiPathAzureChatOpenAI
from uipath_langchain.deepagents import (
    create_uipath_deep_agent,
    with_uipath_managed_workspace,
)


class LaunchRequest(TypedDict):
    """Structured input accepted by the parent workflow."""

    request: str


class LaunchWorkflowState(LaunchRequest):
    """State owned by the parent workflow around the planner node."""

    messages: Annotated[list[AnyMessage], add_messages]
    preflight_complete: bool
    workflow_complete: bool


@tool
def score_launch_readiness(audience: str, constraints: list[str]) -> str:
    """Return a deterministic launch-readiness score for a target audience."""
    score = 80
    if len(constraints) >= 3:
        score -= 10
    if any("compliance" in item.lower() for item in constraints):
        score -= 10
    if any("deadline" in item.lower() for item in constraints):
        score -= 5
    return f"Launch readiness for {audience}: {max(score, 35)}/100"


MODEL = UiPathAzureChatOpenAI(model="gpt-5.4", temperature=0)

PLANNER_PROMPT = """You are the launch-planning node in a larger workflow.

Use score_launch_readiness to assess the requested launch. Use the provided
filesystem tools to write /launch/plan.md before giving a concise plan with the
readiness assessment, milestones, risks, and owners.
"""

launch_planner = create_uipath_deep_agent(
    model=MODEL,
    system_prompt=PLANNER_PROMPT,
    tools=[score_launch_readiness],
)


def preflight(state: LaunchWorkflowState) -> dict[str, bool | list[HumanMessage]]:
    """Adapt the structured request before delegating to the planner node."""
    return {
        "preflight_complete": True,
        "messages": [HumanMessage(content=state["request"])],
    }


def complete(_: LaunchWorkflowState) -> dict[str, bool]:
    """Record that the parent workflow has completed the planner stage."""
    return {"workflow_complete": True}


builder = StateGraph(LaunchWorkflowState, input_schema=LaunchRequest)
builder.add_node("preflight", preflight)
builder.add_node("launch_planner", launch_planner)
builder.add_node("complete", complete)
builder.add_edge(START, "preflight")
builder.add_edge("preflight", "launch_planner")
builder.add_edge("launch_planner", "complete")
builder.add_edge("complete", END)

graph = with_uipath_managed_workspace(builder.compile())
