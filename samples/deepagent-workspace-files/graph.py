"""DeepAgent whose files survive a suspend and resume in an Orchestrator job."""

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.types import interrupt

from uipath_langchain.agent.advanced import UiPathWorkspaceBackend
from uipath_langchain.chat import UiPathChat

REPORT = "report.md"

SYSTEM_PROMPT = f"""You write short reports and revise them after a human review.

1. Write a first draft of the report to /{REPORT} with write_file.
2. Call request_review. The job may suspend here until the reviewer answers.
3. Revise /{REPORT} with edit_file to apply the reviewer's feedback.
4. Reply with the final report."""


backend = UiPathWorkspaceBackend()


@tool
def request_review() -> str:
    """Send the current draft of the report to a human reviewer and return their feedback."""
    report = backend.cwd / REPORT
    draft = report.read_text() if report.exists() else "(no draft found)"
    answer = interrupt({"message": "Review the draft report", "draft": draft})
    if isinstance(answer, dict):
        return str(answer.get("feedback", answer))
    return str(answer)


deep_agent = create_deep_agent(
    model=UiPathChat(model="gpt-4.1-mini-2025-04-14"),
    system_prompt=SYSTEM_PROMPT,
    tools=[request_review],
    backend=backend,
)
