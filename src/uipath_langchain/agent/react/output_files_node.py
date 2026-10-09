"""Verification gate for output file fields, run just before termination.

Sits between the agent loop and TERMINATE, inspecting the pending
``end_execution`` arguments and letting termination proceed only when every
required file field carries a reference to an existing attachment. An accepted
reference is rebuilt from that attachment before termination reads it, and
registered as one of the run's attachments so the agent-output guardrails can
inspect it. A failure answers the tool call with a corrective message and hands
control back to the agent rather than faulting.

Tool *inputs* solve the same problem through ``get_job_attachment_wrapper``,
which rejects an id that is not in ``inner_state.job_attachments``. That wrapper
cannot be reused here for two reasons. It is honored only by ``UiPathToolNode``,
so it never runs on the advanced path, where LangChain executes the tools; and
``end_execution`` is never executed as a tool at all, since routing intercepts
it and reads its arguments directly. Looking each attachment up in Orchestrator
works identically on both paths.
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from uipath.agent.react import END_EXECUTION_TOOL
from uipath.runtime.errors import UiPathErrorCategory

from ..attachments.output_files import (
    OutputFileField,
    check_output_files,
    verified_output_attachments,
)
from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from .types import AgentGraphNode, AgentGraphState
from .utils import extract_current_tool_call_index, find_latest_ai_message


def _pending_end_execution(
    state: AgentGraphState,
) -> tuple[AIMessage, int] | None:
    """The message making the current ``end_execution`` call, and the call's index."""
    last_message = find_latest_ai_message(state.messages)
    if last_message is None or not last_message.tool_calls:
        return None
    index = extract_current_tool_call_index(state.messages)
    if index is None:
        return None
    if last_message.tool_calls[index]["name"] != END_EXECUTION_TOOL.name:
        return None
    return last_message, index


def _with_args(message: AIMessage, index: int, args: dict[str, Any]) -> AIMessage:
    """``message`` with the args of its tool call at ``index`` replaced."""
    tool_calls = list(message.tool_calls)
    tool_calls[index] = {**tool_calls[index], "args": args}
    return message.model_copy(update={"tool_calls": tool_calls})


def create_output_files_node(
    fields: list[OutputFileField],
    max_retries: int,
    file_tool_name: str | None = None,
):
    """Create the node that gates termination on the declared output files."""

    async def output_files_node(
        state: AgentGraphState,
    ) -> Command[Literal[AgentGraphNode.TERMINATE, AgentGraphNode.AGENT]]:
        pending = _pending_end_execution(state)
        if pending is None:
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

        message, index = pending
        tool_call = message.tool_calls[index]
        problem, output = await check_output_files(
            fields, tool_call["args"], file_tool_name
        )
        if problem is None:
            return Command(
                goto=AgentGraphNode.TERMINATE,
                update={
                    "messages": [_with_args(message, index, output)],
                    "inner_state": {
                        "job_attachments": verified_output_attachments(fields, output)
                    },
                },
            )

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
