"""Verification gate for output file fields, run just before termination.

Sits between the agent loop and TERMINATE, inspecting the pending
``end_execution`` arguments and letting termination proceed only when every
required file field carries a reference to an attachment linked to this job. A
failure answers the tool call with a corrective message and hands control back
to the agent rather than faulting.

Tool *inputs* solve the same problem through ``get_job_attachment_wrapper``,
which rejects an id that is not in ``inner_state.job_attachments``. That wrapper
cannot be reused here for two reasons. It is honored only by ``UiPathToolNode``,
so it never runs on the advanced path, where LangChain executes the tools; and
``end_execution`` is never executed as a tool at all, since routing intercepts
it and reads its arguments directly. Checking against the attachments the
platform reports for the job works identically on both paths.
"""

from typing import Literal

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langgraph.types import Command
from uipath.agent.react import END_EXECUTION_TOOL
from uipath.runtime.errors import UiPathErrorCategory

from ..attachments.output_files import OutputFileField, diagnose_output_files
from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from .types import AgentGraphNode, AgentGraphState
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


def create_output_files_node(fields: list[OutputFileField], max_retries: int):
    """Create the node that gates termination on the declared output files."""

    async def output_files_node(
        state: AgentGraphState,
    ) -> Command[Literal[AgentGraphNode.TERMINATE, AgentGraphNode.AGENT]]:
        tool_call = _pending_end_execution(state)
        if tool_call is None:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.ROUTING_ERROR,
                title="Output file verification reached without an end_execution call.",
                detail=(
                    "This node only runs on an end_execution tool call, and the "
                    "router is the only route into it. Passing the output through "
                    "unchecked would skip verification silently."
                ),
                category=UiPathErrorCategory.SYSTEM,
            )

        problem = await diagnose_output_files(fields, tool_call["args"])
        if problem is None:
            return Command(goto=AgentGraphNode.TERMINATE)

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

        return Command(
            goto=AgentGraphNode.AGENT,
            update={
                "messages": [
                    ToolMessage(
                        content=problem,
                        tool_call_id=tool_call["id"],
                        name=END_EXECUTION_TOOL.name,
                        status="error",
                    )
                ],
                "inner_state": {"output_file_retries": retries + 1},
            },
        )

    return output_files_node
