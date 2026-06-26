"""Escalation tool creation for Action Center integration."""

import json
import logging
import os
from enum import Enum
from typing import Any, Literal

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    AgentQuickFormEscalationChannel,
    ArgumentEmailRecipient,
    ArgumentGroupNameRecipient,
    AssetRecipient,
    EscalationChannel,
    LowCodeAgentDefinition,
    StandardRecipient,
)
from uipath.agent.utils.text_tokens import safe_get_nested
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import Task, TaskRecipient, TaskRecipientType
from uipath.platform.common import WaitEscalation
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import (
    get_current_span_and_trace_ids,
    get_execution_folder_path,
)
from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)
from ..react.types import AgentGraphState
from .escalation_memory import (
    EscalationMemorySettings,
    _check_escalation_memory_cache,
    _get_escalation_memory_folder_path,
    _get_escalation_memory_settings,
    _get_escalation_memory_space_id,
    _get_escalation_memory_space_name,
    _ingest_escalation_memory,
    _resolve_user_id,
)
from .tool_node import ToolWrapperReturnType
from .utils import (
    resolve_task_title,
    sanitize_dict_for_serialization,
    sanitize_tool_name,
)

_escalation_logger = logging.getLogger(__name__)


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
    """
    Filter action center task data based on input/output schemas.

    When output_schema is None, returns only fields not present in input_schema.
    When output_schema is provided, returns only fields defined in output_schema.

    Args:
        data: Raw task data from action center
        input_schema: JSON schema defining the input fields
        output_schema: Optional JSON schema defining expected output fields

    Returns:
        Filtered dictionary containing only relevant output fields
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


def _build_escalation_memory_payload(
    serialized_input: dict[str, Any],
    escalation_output: dict[str, Any],
    outcome: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    answer = {"output": escalation_output, "outcome": outcome}
    attributes = {"arguments": serialized_input}
    return answer, attributes


def _pop_escalation_memory_span_context(
    metadata: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    span_context = (metadata or {}).get("_span_context")
    if not isinstance(span_context, dict):
        _escalation_logger.debug(
            "Escalation memory span context missing _span_context metadata"
        )
        return None, None

    parent_span_id = _format_otel_id(span_context.pop("parent_span_id", None), 16)
    trace_id = _format_otel_id(span_context.pop("trace_id", None), 32)
    _escalation_logger.debug(
        "Escalation memory span context: %s",
        json.dumps(
            {
                "parentSpanId": parent_span_id,
                "traceId": trace_id,
                "remainingContext": span_context,
            },
            default=str,
        ),
    )
    return parent_span_id, trace_id


def _format_otel_id(value: Any, width: int) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return f"{value:0{width}x}"
    return str(value)


def _normalize_trace_id(value: str) -> str:
    normalized = value.replace("-", "").lower()
    if len(normalized) != 32:
        raise ValueError(f"Invalid trace ID format: {value}")
    return normalized


def _get_exported_trace_id(trace_id: str | None) -> str | None:
    trace_id_override = os.environ.get("UIPATH_TRACE_ID")
    if trace_id_override:
        try:
            return _normalize_trace_id(trace_id_override)
        except ValueError:
            _escalation_logger.warning(
                "Ignoring invalid UIPATH_TRACE_ID override: %s",
                trace_id_override,
            )

    return trace_id


def _try_get_channel_app_name(channel: EscalationChannel) -> str | None:
    return (
        channel.properties.app_name
        if isinstance(channel, AgentEscalationChannel)
        else None
    )


async def create_task_for_channel(
    client: UiPath,
    channel: EscalationChannel,
    *,
    title: str,
    data: dict[str, Any],
    recipient: TaskRecipient | None,
    folder_path: str | None,
) -> Task:
    """Create the human task backing an escalation channel."""
    if isinstance(channel, AgentQuickFormEscalationChannel):
        schema_id = channel.properties.schema_id
        assert schema_id is not None
        return await client.tasks.create_quickform_async(
            title=title,
            task_schema_key=schema_id,
            schema=channel.properties.form_schema,
            data=data,
            folder_path=folder_path,
            recipient=recipient,
            priority=channel.priority,
            labels=channel.labels,
            is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
            actionable_message_metadata=channel.properties.actionable_message_meta_data,
        )
    return await client.tasks.create_async(
        title=title,
        data=data,
        app_name=channel.properties.app_name,
        app_folder_path=folder_path,
        recipient=recipient,
        priority=channel.priority,
        labels=channel.labels,
        is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
        actionable_message_metadata=channel.properties.actionable_message_meta_data,
    )


def _resolve_channel(resource: AgentEscalationResourceConfig) -> EscalationChannel:
    """Return the escalation's channel, validating quick-form configuration."""
    channel = resource.channels[0]
    if (
        isinstance(channel, AgentQuickFormEscalationChannel)
        and channel.properties.schema_id is None
    ):
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Quick form escalation is missing a schema id",
            detail=(
                f"Escalation '{channel.name}' has a quick form "
                "schema without a schemaId."
            ),
            category=UiPathErrorCategory.USER,
        )
    return channel


def create_escalation_tool(
    resource: AgentEscalationResourceConfig,
    agent: LowCodeAgentDefinition | None = None,
) -> StructuredTool:
    """Build the human-in-the-loop escalation tool for an escalation resource."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: EscalationChannel = _resolve_channel(resource)

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)

    class EscalationToolOutput(BaseModel):
        action: Literal["approve", "reject"]
        data: output_model
        is_deleted: bool = False

    _span_context: dict[str, Any] = {}
    _bts_context: dict[str, Any] = {}
    _memory_space_id: str | None = _get_escalation_memory_space_id(resource, agent)
    _memory_folder_path: str | None = _get_escalation_memory_folder_path(
        resource, agent
    )
    _memory_space_name: str | None = _get_escalation_memory_space_name(resource, agent)
    _memory_settings: EscalationMemorySettings | None = _get_escalation_memory_settings(
        resource
    )

    async def escalation_tool_fn(**kwargs: Any) -> dict[str, Any]:
        agent_input: dict[str, Any] = (
            tool.metadata.get("agent_input") if tool.metadata else None
        ) or {}
        recipient: TaskRecipient | None = (
            await resolve_recipient_value(channel.recipients[0], input_args=agent_input)
            if channel.recipients
            else None
        )
        folder_path = get_execution_folder_path()

        task_title = "Escalation Task"
        if tool.metadata is not None:
            # Recipient requires runtime resolution, store in metadata after resolving
            tool.metadata["recipient"] = recipient
            task_title = tool.metadata.get("task_title") or task_title

        serialized_data = input_model.model_validate(kwargs).model_dump(mode="json")

        # --- Escalation memory: check cache before creating HITL task ---
        if _memory_space_id:
            cached_result = await _check_escalation_memory_cache(
                _memory_space_id,
                serialized_data,
                folder_path=_memory_folder_path or folder_path,
                memory_settings=_memory_settings,
                memory_space_name=_memory_space_name,
            )
            if cached_result is not None:
                return {
                    "action": _resolve_escalation_action(
                        cached_result.outcome,
                        channel.outcome_mapping,
                    ),
                    "output": cached_result.output,
                    "outcome": cached_result.outcome,
                }

        @mockable(
            name=tool_name.lower(),
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=EscalationToolOutput.model_json_schema(),
            example_calls=channel.properties.example_calls,
        )
        async def escalate(**_tool_kwargs: Any):
            @durable_interrupt
            async def create_escalation_task():
                client = UiPath()
                created_task = await create_task_for_channel(
                    client,
                    channel,
                    title=task_title,
                    data=serialized_data,
                    recipient=recipient,
                    folder_path=folder_path,
                )

                if created_task.id is not None:
                    _bts_context["task_key"] = str(created_task.id)

                return WaitEscalation(
                    action=created_task,
                    app_folder_path=folder_path,
                    app_name=_try_get_channel_app_name(channel),
                    recipient=recipient,
                )

            return await create_escalation_task()

        result = await escalate(**kwargs)
        if isinstance(result, dict):
            result = Task.model_validate(result)

        if result.is_deleted:
            return {
                "action": EscalationAction.END,
                "output": None,
                "outcome": "The escalation task was deleted",
            }

        outcome = result.action
        escalation_output = _parse_task_data(
            result.data.model_dump()
            if isinstance(result.data, BaseModel)
            else result.data,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
        )

        escalation_action = _resolve_escalation_action(
            outcome,
            channel.outcome_mapping,
        )

        # --- Escalation memory: persist outcome for future recall ---
        if _memory_space_id:
            user_id = await _resolve_user_id(result.completed_by_user)
            parent_span_id, trace_id = _pop_escalation_memory_span_context(
                tool.metadata
            )
            if not parent_span_id or not trace_id:
                fallback_span_id, fallback_trace_id = get_current_span_and_trace_ids()
                _escalation_logger.debug(
                    "Escalation memory span context fallback: %s",
                    json.dumps(
                        {
                            "fallbackSpanId": fallback_span_id,
                            "fallbackTraceId": fallback_trace_id,
                            "hadParentSpanId": bool(parent_span_id),
                            "hadTraceId": bool(trace_id),
                        },
                        default=str,
                    ),
                )
                parent_span_id = parent_span_id or fallback_span_id
                trace_id = trace_id or _get_exported_trace_id(fallback_trace_id)
            if not parent_span_id or not trace_id:
                _escalation_logger.warning(
                    "Skipping escalation memory ingest because span provenance is incomplete"
                )
                return {
                    "action": escalation_action,
                    "output": escalation_output,
                    "outcome": outcome,
                }
            answer_payload, attributes_payload = _build_escalation_memory_payload(
                serialized_data,
                escalation_output,
                outcome,
            )
            await _ingest_escalation_memory(
                _memory_space_id,
                answer=json.dumps(answer_payload),
                attributes=json.dumps(attributes_payload),
                parent_span_id=parent_span_id,
                trace_id=trace_id,
                user_id=user_id,
                folder_path=_memory_folder_path or folder_path,
                memory_space_name=_memory_space_name,
            )
            if user_id is None:
                _escalation_logger.info(
                    "Ingested escalation memory without reviewer user ID "
                    "because the completed user could not be resolved"
                )

        return {
            "action": escalation_action,
            "output": escalation_output,
            "outcome": outcome,
        }

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
            default_title="Escalation Task",
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

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        output_type=output_model,
        coroutine=escalation_tool_fn,
        argument_properties=channel.argument_properties,
        metadata={
            "tool_type": "escalation",
            "display_name": _try_get_channel_app_name(channel) or channel.name,
            "channel_type": channel.type,
            "recipient": None,
            "args_schema": input_model,
            "output_schema": output_model,
            "_span_context": _span_context,
            "_bts_context": _bts_context,
        },
    )
    tool.set_tool_wrappers(awrapper=escalation_wrapper)

    return tool
