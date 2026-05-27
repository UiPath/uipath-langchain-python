"""Shared primitives for Action Center escalation tools.

This module is the seam between the per-variant escalation factories
(``app_task.py``, ``quick_form.py``, ``ixp_vs.py``) and the tool layer.
It owns:

* The escalation outcome model (:class:`EscalationAction`,
  :func:`make_escalation_tool_output`).
* Recipient and asset resolution.
* Output post-processing (:func:`_parse_task_data`,
  :func:`_resolve_escalation_action`).
* The invocation preamble shared by every escalation variant
  (:class:`EscalationInvocationCtx`, :func:`build_invocation_ctx`).
* The post-interrupt finaliser (:func:`finalize_escalation_result`).
* The LangGraph tool wrapper factory (:func:`make_escalation_wrapper`).

The escalation factories assemble these primitives — they no longer
duplicate the scaffolding.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import create_model as pydantic_create_model
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    ArgumentEmailRecipient,
    ArgumentGroupNameRecipient,
    AssetRecipient,
    StandardRecipient,
)
from uipath.agent.utils.text_tokens import safe_get_nested
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import get_execution_folder_path

from ...exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ...react.types import AgentGraphState
from ..tool_node import ToolWrapperReturnType
from ..utils import resolve_task_title, sanitize_dict_for_serialization


class EscalationAction(str, Enum):
    """Actions that can be taken after an escalation completes."""

    CONTINUE = "continue"
    END = "end"


async def resolve_recipient_value(
    recipient: AgentEscalationRecipient,
    input_args: dict[str, Any] | None = None,
) -> TaskRecipient | None:
    """Resolve recipient value based on recipient type."""
    if isinstance(recipient, AssetRecipient):
        value = await resolve_asset(recipient.asset_name, get_execution_folder_path())
        type = None
        if recipient.type == AgentEscalationRecipientType.ASSET_USER_EMAIL:
            type = TaskRecipientType.EMAIL
        elif recipient.type == AgentEscalationRecipientType.ASSET_GROUP_NAME:
            type = TaskRecipientType.GROUP_NAME
        return TaskRecipient(value=value, type=type, displayName=value)

    if isinstance(recipient, ArgumentEmailRecipient):
        value = safe_get_nested(input_args or {}, recipient.argument_path)
        if value is None:
            raise ValueError(
                f"Argument '{recipient.argument_path}' has no value in agent input."
            )
        return TaskRecipient(
            value=value, type=TaskRecipientType.EMAIL, displayName=value
        )

    if isinstance(recipient, ArgumentGroupNameRecipient):
        value = safe_get_nested(input_args or {}, recipient.argument_path)
        if value is None:
            raise ValueError(
                f"Argument '{recipient.argument_path}' has no value in agent input."
            )
        return TaskRecipient(
            value=value, type=TaskRecipientType.GROUP_NAME, displayName=value
        )

    if isinstance(recipient, StandardRecipient):
        type = TaskRecipientType(recipient.type)
        if recipient.type == AgentEscalationRecipientType.USER_EMAIL:
            type = TaskRecipientType.EMAIL
        return TaskRecipient(
            value=recipient.value, type=type, displayName=recipient.value
        )

    return None


async def resolve_asset(asset_name: str, folder_path: str | None) -> str | None:
    """Retrieve asset value."""
    try:
        client = UiPath()
        result = await client.assets.retrieve_async(
            name=asset_name, folder_path=folder_path
        )

        if not result or not result.value:
            raise ValueError(f"Asset '{asset_name}' has no value configured.")

        return result.value
    except Exception as e:
        raise ValueError(
            f"Failed to resolve asset '{asset_name}' in folder '{folder_path}': {str(e)}"
        ) from e


def _parse_task_data(
    data: dict[str, Any],
    input_schema: dict[str, Any],
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter action center task data based on input/output schemas.

    When output_schema is None, returns only fields not present in input_schema.
    When output_schema is provided, returns only fields defined in output_schema.
    """
    filtered_fields: dict[str, Any] = {}

    if output_schema is None:
        input_field_names = set()
        if "properties" in input_schema:
            input_field_names = set(input_schema["properties"].keys())

        for field_name, field_value in data.items():
            if field_name not in input_field_names:
                filtered_fields[field_name] = field_value

    else:
        output_field_names = set()
        if "properties" in output_schema:
            output_field_names = set(output_schema["properties"].keys())

        for field_name, field_value in data.items():
            if field_name in output_field_names:
                filtered_fields[field_name] = field_value

    return filtered_fields


def _resolve_escalation_action(
    outcome: str | None,
    outcome_mapping: dict[str, str] | None,
) -> EscalationAction:
    outcome_action = (
        outcome_mapping.get(outcome) if outcome_mapping and outcome else None
    )
    return (
        EscalationAction(outcome_action)
        if outcome_action
        else EscalationAction.CONTINUE
    )


def make_escalation_tool_output(output_model: Any) -> type[BaseModel]:
    """Build the escalation tool output schema for a given output model.

    Every escalation variant returns ``{action, data, is_deleted}`` to the
    mockable layer. The ``data`` field is parameterised by the channel's
    output model.
    """
    return pydantic_create_model(
        "EscalationToolOutput",
        action=(Literal["approve", "reject"], ...),
        data=(output_model, ...),
        is_deleted=(bool, False),
    )


@dataclass
class EscalationInvocationCtx:
    """Per-invocation data assembled before opening the durable interrupt."""

    agent_input: dict[str, Any]
    recipient: TaskRecipient | None
    folder_path: str | None
    task_title: str
    serialized_data: dict[str, Any]


async def build_invocation_ctx(
    tool: BaseTool,
    channel: AgentEscalationChannel,
    kwargs: dict[str, Any],
    input_model: Any,
    *,
    default_title: str = "Escalation Task",
) -> EscalationInvocationCtx:
    """Assemble the preamble every escalation variant runs.

    Resolves the recipient, captures the execution folder, picks up
    the wrapper-resolved task title from ``tool.metadata``, and
    validates the input payload into a JSON-mode dict.
    """
    agent_input: dict[str, Any] = (
        tool.metadata.get("agent_input") if tool.metadata else None
    ) or {}
    recipient: TaskRecipient | None = (
        await resolve_recipient_value(channel.recipients[0], input_args=agent_input)
        if channel.recipients
        else None
    )
    folder_path = get_execution_folder_path()

    task_title = default_title
    if tool.metadata is not None:
        # The wrapper resolves recipient and title; persist them so the
        # nested durable_interrupt closure can read them back.
        tool.metadata["recipient"] = recipient
        task_title = tool.metadata.get("task_title") or default_title

    serialized_data = input_model.model_validate(kwargs).model_dump(mode="json")
    return EscalationInvocationCtx(
        agent_input=agent_input,
        recipient=recipient,
        folder_path=folder_path,
        task_title=task_title,
        serialized_data=serialized_data,
    )


def finalize_escalation_result(
    result: Any,
    *,
    input_model: Any,
    output_model: Any,
    outcome_mapping: dict[str, str] | None,
) -> dict[str, Any]:
    """Post-process the action center task into the tool's response shape.

    Returns ``{action, output, outcome}`` where ``action`` is an
    :class:`EscalationAction`. Handles the deleted-task short-circuit
    so callers do not have to repeat it.
    """
    if result.is_deleted:
        return {
            "action": EscalationAction.END,
            "output": None,
            "outcome": "The escalation task was deleted",
        }

    outcome = result.action
    raw_data = (
        result.data.model_dump()
        if isinstance(result.data, BaseModel)
        else (result.data or {})
    )
    escalation_output = _parse_task_data(
        raw_data,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    escalation_action = _resolve_escalation_action(outcome, outcome_mapping)
    return {
        "action": escalation_action,
        "output": escalation_output,
        "outcome": outcome,
    }


def make_escalation_wrapper(
    channel: AgentEscalationChannel,
    *,
    default_title: str = "Escalation Task",
):
    """Build the LangGraph tool wrapper for an escalation channel.

    The wrapper resolves the task title from agent state, captures the
    call's id and args into ``tool.metadata`` for downstream readers
    (escalation memory, observability), invokes the tool, and raises
    :class:`AgentRuntimeError` with code
    ``TERMINATION_ESCALATION_REJECTED`` when the tool resolves to
    :attr:`EscalationAction.END`.
    """

    async def escalation_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        if tool.metadata is None:
            raise RuntimeError("Tool metadata is required for task_title resolution")

        state_dict = sanitize_dict_for_serialization(dict(state))
        tool.metadata["task_title"] = resolve_task_title(
            channel.task_title,
            state_dict,
            default_title=default_title,
        )
        internal_fields = set(AgentGraphState.model_fields.keys())
        tool.metadata["agent_input"] = {
            k: v for k, v in state_dict.items() if k not in internal_fields
        }

        tool.metadata["_call_id"] = call.get("id")
        tool.metadata["_call_args"] = dict(call.get("args", {}))

        result = await tool.ainvoke(call["args"])

        if result["action"] == EscalationAction.END:
            output_detail = f"Escalation output: {result['output']}"
            termination_title = (
                f"Agent run ended based on escalation outcome {result['action']} "
                f"with directive {result['outcome']}"
            )
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.TERMINATION_ESCALATION_REJECTED,
                title=termination_title,
                detail=output_detail,
                category=UiPathErrorCategory.USER,
            )

        return {
            "output": result["output"],
            "outcome": result["outcome"],
            "task_id": result.get("task_id"),
            "assigned_to": result.get("assigned_to"),
        }

    return escalation_wrapper
