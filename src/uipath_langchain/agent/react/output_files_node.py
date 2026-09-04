"""Verification gate for output file fields, run just before termination.

Sits between the agent loop and TERMINATE whenever the output schema declares a
job-attachment field. It inspects the pending ``end_execution`` arguments and
lets termination proceed only when every required file field carries a reference
to an attachment that is actually linked to this job.

A failure is not fatal. The node answers the ``end_execution`` tool call with a
corrective message and hands control back to the agent, which can create the
missing file and end again. Only a run that keeps failing that check faults, so
a model that forgets the tool costs a turn rather than the job.
"""

from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from uipath.agent.react import END_EXECUTION_TOOL
from uipath.runtime.errors import UiPathErrorCategory

from ..attachments.output_files import OutputFileField, diagnose_output_files
from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from .types import AgentGraphState
from .utils import extract_current_tool_call_index, find_latest_ai_message


def _pending_end_execution(state: AgentGraphState) -> ToolCall | None:
    """The ``end_execution`` tool call the agent is currently making, if any."""
    last_message = find_latest_ai_message(state.messages)
    if last_message is None or not last_message.tool_calls:
        return None
    index = extract_current_tool_call_index(state.messages)
    if index is None:
        return None
    tool_call = last_message.tool_calls[index]
    if tool_call["name"] != END_EXECUTION_TOOL.name:
        return None
    return tool_call


def _cleared() -> dict[str, Any]:
    """State update that records a passing verification."""
    return {"inner_state": {"output_file_problem": None}}


def create_output_files_node(fields: list[OutputFileField], max_retries: int):
    """Create the node that gates termination on the declared output files."""

    async def output_files_node(state: AgentGraphState) -> dict[str, Any]:
        tool_call = _pending_end_execution(state)
        if tool_call is None:
            return _cleared()

        problem = await diagnose_output_files(fields, tool_call["args"])
        if problem is None:
            return _cleared()

        retries = state.inner_state.output_file_retries
        if retries >= max_retries:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.OUTPUT_VALIDATION_ERROR,
                title="Agent did not produce the required output file",
                detail=(
                    f"{problem} The agent was given {max_retries} chance(s) to "
                    "correct this and did not. Verify the agent's prompt asks for "
                    "the file, and that the output schema's file fields are the "
                    "ones you intend."
                ),
                category=UiPathErrorCategory.USER,
            )

        return {
            "messages": [
                ToolMessage(
                    content=problem,
                    tool_call_id=tool_call["id"],
                    name=END_EXECUTION_TOOL.name,
                    status="error",
                )
            ],
            "inner_state": {
                "output_file_retries": retries + 1,
                "output_file_problem": problem,
            },
        }

    return output_files_node


def output_files_verified(state: AgentGraphState) -> bool:
    """Whether the last verification pass let termination proceed."""
    return state.inner_state.output_file_problem is None
