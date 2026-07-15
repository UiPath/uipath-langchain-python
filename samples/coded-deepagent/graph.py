"""Task-mode coded DeepAgent using the standard advanced-agent graph builder."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from uipath_langchain.agent.advanced import create_advanced_agent_graph
from uipath_langchain.chat import UiPathChat


class BriefInput(BaseModel):
    """Input for the product launch brief."""

    product_name: str = Field(description="Name of the product or feature.")
    audience: str = Field(description="Primary customer or user audience.")
    objective: str = Field(description="Main launch objective.")
    constraints: list[str] = Field(
        default_factory=list,
        description="Important limits, requirements, or risks to account for.",
    )


class BriefOutput(BaseModel):
    """Structured launch brief returned by the agent."""

    executive_summary: str
    launch_plan: list[str]
    risk_review: list[str]
    workspace_files: list[str]


@tool
def score_launch_readiness(audience: str, constraints: list[str]) -> str:
    """Score launch readiness from simple deterministic planning signals."""
    score = 80
    if len(constraints) >= 3:
        score -= 10
    if any("compliance" in item.lower() for item in constraints):
        score -= 10
    if any("deadline" in item.lower() for item in constraints):
        score -= 5
    return f"Launch readiness for {audience}: {max(score, 35)}/100"


MODEL = UiPathChat(model="gpt-4o-2024-08-06", temperature=0)

RISK_REVIEWER_PROMPT = """You are a launch risk reviewer.
Review the proposed launch plan for practical delivery risks, compliance gaps,
unclear ownership, and missing follow-up work. Return concise findings that the
main agent can incorporate into the final brief."""

RISK_REVIEWER = {
    "name": "risk_reviewer",
    "description": "Reviews launch plans for execution risks and missing safeguards.",
    "system_prompt": RISK_REVIEWER_PROMPT,
    "model": MODEL,
}


def build_system_prompt(args: dict) -> str:
    return f"""You are a product launch planning agent for {args["product_name"]}.

Use the planning tools provided by DeepAgents. Use score_launch_readiness to get
a deterministic readiness signal. Delegate a plan review to risk_reviewer before
finalizing the answer.

Tailor all planning to this audience: {args["audience"]}.

The UiPath runtime detects this DeepAgents graph and provides its filesystem.
Write these workspace files before producing the final answer:
- /launch/brief.md with the final launch brief
- /launch/risks.md with the risk review

Return structured output matching the schema. Include the workspace file paths
you wrote in workspace_files."""


def build_user_prompt(args: dict) -> str:
    constraints = args.get("constraints") or []
    constraint_lines = "\n".join(f"- {item}" for item in constraints) or "- None"
    return f"""Create a launch brief.

Product: {args["product_name"]}
Audience: {args["audience"]}
Objective: {args["objective"]}
Constraints:
{constraint_lines}
"""


graph = create_advanced_agent_graph(
    model=MODEL,
    input_schema=BriefInput,
    output_schema=BriefOutput,
    system_prompt=build_system_prompt,
    build_user_message=build_user_prompt,
    tools=[score_launch_readiness],
    subagents=[RISK_REVIEWER],
)
