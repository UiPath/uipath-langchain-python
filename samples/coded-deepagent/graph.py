"""Task-mode coded DeepAgent using the standard DeepAgents API."""

from deepagents import create_deep_agent
from langchain_core.tools import tool
from pydantic import BaseModel

from uipath_langchain.chat import UiPathChat


class BriefOutput(BaseModel):
    """Structured launch brief returned by the agent."""

    executive_summary: str
    launch_plan: list[str]
    risk_review: list[str]


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


SYSTEM_PROMPT = """You are a product launch planning agent.

Use the planning tools provided by DeepAgents. Use score_launch_readiness to get
a deterministic readiness signal. Delegate a plan review to risk_reviewer before
finalizing the answer.

Use the DeepAgents filesystem to write these working files before producing the
final answer:
- /launch/brief.md with the final launch brief
- /launch/risks.md with the risk review

Return structured output matching the schema."""


graph = create_deep_agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    response_format=BriefOutput,
    tools=[score_launch_readiness],
    subagents=[RISK_REVIEWER],
)
