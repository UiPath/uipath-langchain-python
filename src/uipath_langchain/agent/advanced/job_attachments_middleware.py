"""Attachment references in tool calls for agents whose loop LangChain runs.

The standard graph checks a tool call's attachment references in
``get_job_attachment_wrapper``, which only ``UiPathToolNode`` honors. An advanced
agent's tools run in LangChain's ``ToolNode``, so this middleware does the same
work around each call: it resolves every reference against the attachments the
run has seen, falls back to Orchestrator for the rest, and remembers what each
tool returns.

The known attachments live in graph state rather than on the middleware, so they
survive a suspend and resume, and so deepagents copies them into a subagent and
back out of it, which it does for every state key not marked private.
"""

from typing import Annotated, Any, Awaitable, Callable, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.attachments.job_attachments import (
    get_job_attachment_paths,
    get_job_attachments,
    parse_attachments_from_conversation_messages,
    parse_tool_content,
    resolve_attachment_references,
)
from uipath_langchain.agent.attachments.pydantic_json import coerce_json_strings

JOB_ATTACHMENTS_STATE_KEY = "uipath__job_attachments"


def _merge_attachments(
    left: dict[str, dict[str, Any]] | None, right: dict[str, dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    return {**(left or {}), **(right or {})}


class JobAttachmentsState(AgentState[Any]):
    uipath__job_attachments: NotRequired[
        Annotated[dict[str, dict[str, Any]], _merge_attachments]
    ]


def dump_attachments(attachments: dict[str, Attachment]) -> dict[str, dict[str, Any]]:
    """Attachments as the JSON-safe state value, keyed by id."""
    return {
        id: attachment.model_dump(by_alias=True, mode="json")
        for id, attachment in attachments.items()
    }


def _load_attachments(value: dict[str, dict[str, Any]] | None) -> dict[str, Attachment]:
    return {
        id: Attachment.model_validate(attachment)
        for id, attachment in (value or {}).items()
    }


def _args_schema(request: ToolCallRequest) -> type[BaseModel] | None:
    schema = getattr(request.tool, "args_schema", None)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    return None


def _produced_attachments(
    request: ToolCallRequest, result: ToolMessage | Command[Any]
) -> dict[str, Attachment]:
    output_type = getattr(request.tool, "output_type", None)
    if not isinstance(result, ToolMessage) or not (
        isinstance(output_type, type) and issubclass(output_type, BaseModel)
    ):
        return {}
    return {
        str(attachment.id): attachment
        for attachment in get_job_attachments(
            output_type, parse_tool_content(result.content)
        )
        if attachment.id is not None
    }


def _with_attachments(
    result: ToolMessage | Command[Any], attachments: dict[str, Attachment]
) -> ToolMessage | Command[Any]:
    update = {JOB_ATTACHMENTS_STATE_KEY: dump_attachments(attachments)}
    if isinstance(result, ToolMessage):
        return Command(update={"messages": [result], **update})
    if isinstance(result.update, dict):
        return Command(
            graph=result.graph,
            goto=result.goto,
            resume=result.resume,
            update={**result.update, **update},
        )
    return result


class JobAttachmentsMiddleware(AgentMiddleware[JobAttachmentsState, Any]):
    """Resolve and remember attachment references in an advanced agent's tool calls."""

    state_schema = JobAttachmentsState

    async def abefore_agent(
        self, state: JobAttachmentsState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        attached = parse_attachments_from_conversation_messages(state["messages"])
        if not attached:
            return None
        return {JOB_ATTACHMENTS_STATE_KEY: dump_attachments(attached)}

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        schema = _args_schema(request)
        found: dict[str, Attachment] = {}
        if schema is not None and (paths := get_job_attachment_paths(schema)):
            known = _load_attachments(request.state.get(JOB_ATTACHMENTS_STATE_KEY))
            resolution = await resolve_attachment_references(
                paths, call["args"], known, lookup_unknown=True
            )
            if resolution.errors:
                return ToolMessage(
                    content="\n".join(resolution.errors),
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            found = resolution.found
            request = request.override(
                tool_call={**call, "args": coerce_json_strings(resolution.args, schema)}
            )

        result = await handler(request)
        attachments = {**found, **_produced_attachments(request, result)}
        return _with_attachments(result, attachments) if attachments else result
