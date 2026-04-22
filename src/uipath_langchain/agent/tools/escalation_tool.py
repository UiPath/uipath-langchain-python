"""Escalation tool creation for Action Center integration."""

import json
import logging
import os
from enum import Enum
from typing import Any, Literal

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, TypeAdapter
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    ArgumentEmailRecipient,
    ArgumentGroupNameRecipient,
    AssetRecipient,
    StandardRecipient,
)
from uipath.agent.utils.text_tokens import safe_get_nested
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath.platform.common import WaitEscalation
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from ..react.types import AgentGraphState
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


def _get_user_email(user: Any) -> str | None:
    """Extract email from user object/dict."""
    if user is None:
        return None
    if isinstance(user, dict):
        return user.get("emailAddress")
    return getattr(user, "emailAddress", None)


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


def _get_escalation_memory_space_id(
    resource: AgentEscalationResourceConfig,
) -> str | None:
    """Resolve memory space ID from escalation resource extra fields."""
    if not resource.is_agent_memory_enabled:
        return None
    return getattr(resource, "memorySpaceId", None) or getattr(
        resource, "memory_space_id", None
    )


def _get_escalation_memory_settings(
    resource: AgentEscalationResourceConfig,
) -> dict[str, Any] | None:
    """Extract memory settings from escalation resource properties.

    Maps to EscalationResourceDefinition.Properties.Memory in the Temporal
    backend (backend/Common.Models/AgentExecution/ResourceDefinition.cs:96).
    """
    if not resource.is_agent_memory_enabled:
        return None
    props = getattr(resource, "properties", None)
    if isinstance(props, dict):
        return props.get("memory")
    if props is not None:
        return getattr(props, "memory", None)
    return None


def create_escalation_tool(
    resource: AgentEscalationResourceConfig,
) -> StructuredTool:
    """Uses interrupt() for Action Center human-in-the-loop."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)

    class EscalationToolOutput(BaseModel):
        action: Literal["approve", "reject"]
        data: output_model
        is_deleted: bool = False

    _bts_context: dict[str, Any] = {}
    _memory_space_id: str | None = _get_escalation_memory_space_id(resource)
    _memory_settings: dict[str, Any] | None = _get_escalation_memory_settings(resource)

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
        cached_result = await _check_escalation_memory_cache(
            _memory_space_id,
            serialized_data,
            folder_path=folder_path,
            memory_settings=_memory_settings,
        )
        if cached_result is not None:
            return cached_result

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
                created_task = await client.tasks.create_async(
                    title=task_title,
                    data=serialized_data,
                    app_name=channel.properties.app_name,
                    app_folder_path=folder_path,
                    recipient=recipient,
                    priority=channel.priority,
                    labels=channel.labels,
                    is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
                    actionable_message_metadata=channel.properties.actionable_message_meta_data,
                )

                if created_task.id is not None:
                    _bts_context["task_key"] = str(created_task.id)

                return WaitEscalation(
                    action=created_task,
                    app_folder_path=folder_path,
                    app_name=channel.properties.app_name,
                    recipient=recipient,
                )

            return await create_escalation_task()

        result = await escalate(**kwargs)
        if isinstance(result, dict):
            result = TypeAdapter(EscalationToolOutput).validate_python(result)

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

        outcome_str = (
            channel.outcome_mapping.get(outcome)
            if channel.outcome_mapping and outcome
            else None
        )
        escalation_action = (
            EscalationAction(outcome_str) if outcome_str else EscalationAction.CONTINUE
        )

        # --- Escalation memory: persist outcome for future recall ---
        # Shape must match Temporal backend (EscalationToolExecutor.cs):
        #   answer:     new { taskResult.Output, taskResult.Outcome }         (line 485)
        #   attributes: new JsonObject { ["arguments"] = payload.Input.Arguments } (line 503)
        #   spanId/traceId/userId: lines 522-526
        span_id, trace_id = _get_current_span_and_trace_ids()
        await _ingest_escalation_memory(
            _memory_space_id,
            answer=json.dumps({"output": escalation_output, "outcome": outcome}),
            attributes=json.dumps({"arguments": serialized_data}),
            span_id=span_id,
            trace_id=trace_id,
            folder_path=folder_path,
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
            "display_name": channel.properties.app_name,
            "channel_type": channel.type,
            "recipient": None,
            "args_schema": input_model,
            "output_schema": output_model,
            "_bts_context": _bts_context,
        },
    )
    tool.set_tool_wrappers(awrapper=escalation_wrapper)

    return tool


# --- Escalation memory helpers ---


def _get_current_span_and_trace_ids() -> tuple[str, str]:
    """Get current OpenTelemetry span ID and trace ID.

    Returns hex-encoded IDs, or empty strings if no active span.
    Mirrors Temporal backend: payload.SpanId and toolCall.Metadata.Traces.TraceId
    (EscalationToolExecutor.cs lines 522-523).
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            return (
                format(ctx.span_id, "016x"),
                format(ctx.trace_id, "032x"),
            )
    except ImportError:
        pass

    # Fall back to env var for trace_id
    trace_id = os.environ.get("UIPATH_TRACE_ID", "")
    return ("", trace_id)


def _set_memory_span_attribute(name: str, value: bool) -> None:
    """Set a memory-related attribute on the current OTel span.

    Mirrors EscalationToolSpanAttributes in the Temporal backend
    (EscalationToolSpanAttributes.cs:19-25, EscalationToolWorkflow.cs:131-133).
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(name, value)
    except ImportError:
        pass


async def _check_escalation_memory_cache(
    memory_space_id: str | None,
    serialized_input: dict[str, Any],
    folder_path: str | None = None,
    memory_settings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Check escalation memory for a cached answer.

    SearchSettings (threshold, searchMode) are read from the user's memory
    settings on the escalation resource, matching the Temporal backend's
    BuildMemorySearchRequest (EscalationToolExecutor.cs:714-747).
    result_count is always 1 for escalation memory.

    Returns the cached result dict if found, None otherwise.
    """
    if not memory_space_id:
        return None

    try:
        from uipath.platform.memory import (
            FieldSettings,
            MemorySearchRequest,
            SearchField,
            SearchMode,
            SearchSettings,
        )

        # Read search settings from user's memory config (threshold, searchMode),
        # falling back to defaults. result_count is always 1 for escalation memory.
        # Ref: EscalationToolExecutor.cs BuildMemorySearchRequest (lines 740-743)
        threshold = 0.0
        search_mode = SearchMode.Hybrid
        field_settings_lookup: dict[str, dict[str, Any]] = {}
        if memory_settings:
            threshold = memory_settings.get("threshold", 0.0)
            mode_str = memory_settings.get("searchMode", "Hybrid")
            search_mode = (
                SearchMode(mode_str) if mode_str in SearchMode.__members__ else SearchMode.Hybrid
            )
            for fs in memory_settings.get("fieldSettings", []):
                if isinstance(fs, dict) and "name" in fs:
                    field_settings_lookup[fs["name"]] = fs

        fields: list[SearchField] = []
        for k, v in serialized_input.items():
            if v is None:
                continue
            # When field settings are configured, only include fields with
            # configured weights (matching Temporal backend behavior)
            if field_settings_lookup and k not in field_settings_lookup:
                continue
            settings = FieldSettings()
            if k in field_settings_lookup:
                fs = field_settings_lookup[k]
                settings = FieldSettings(weight=fs.get("weight", 1.0))
            fields.append(SearchField(key_path=[k], value=str(v), settings=settings))
        if not fields:
            return None

        request = MemorySearchRequest(
            fields=fields,
            settings=SearchSettings(
                threshold=threshold,
                result_count=1,
                search_mode=search_mode,
            ),
        )
        sdk = UiPath()
        folder_key = (
            sdk.folders.retrieve_folder_key(folder_path) if folder_path else None
        )
        response = await sdk.memory.escalation_search_async(
            memory_space_id=memory_space_id,
            request=request,
            folder_key=folder_key,
        )
        if response.results and response.results[0].answer:
            cached = response.results[0].answer
            _escalation_logger.info(
                "Escalation memory cache hit for space '%s'", memory_space_id
            )
            # Ref: EscalationToolWorkflow.cs:103 — span.Attributes.FromMemory = true
            _set_memory_span_attribute("fromMemory", True)
            return {
                "action": EscalationAction.CONTINUE,
                "output": cached.output,
                "outcome": cached.outcome or "cached",
            }
    except Exception:
        _escalation_logger.warning(
            "Escalation memory search failed for space '%s'",
            memory_space_id,
            exc_info=True,
        )

    return None


async def _ingest_escalation_memory(
    memory_space_id: str | None,
    answer: str,
    attributes: str,
    span_id: str = "",
    trace_id: str = "",
    folder_path: str | None = None,
) -> None:
    """Persist a resolved escalation outcome into memory.

    Sets span attributes to track memory state (EscalationToolWorkflow.cs:131-133):
      fromMemory=false (result was not from cache), savedToMemory=true/false.
    """
    if not memory_space_id:
        return

    # Ref: EscalationToolWorkflow.cs:132 — span.Attributes.FromMemory = false
    _set_memory_span_attribute("fromMemory", False)

    try:
        from uipath.platform.memory import EscalationMemoryIngestRequest

        request = EscalationMemoryIngestRequest(
            span_id=span_id or "unknown",
            trace_id=trace_id or "unknown",
            answer=answer,
            attributes=attributes,
        )
        sdk = UiPath()
        folder_key = (
            sdk.folders.retrieve_folder_key(folder_path) if folder_path else None
        )
        await sdk.memory.escalation_ingest_async(
            memory_space_id=memory_space_id,
            request=request,
            folder_key=folder_key,
        )
        # Ref: EscalationToolExecutor.cs:543 — savedToMemory = true on success
        _set_memory_span_attribute("savedToMemory", True)
        _escalation_logger.info(
            "Ingested escalation outcome into memory space '%s'", memory_space_id
        )
    except Exception:
        _set_memory_span_attribute("savedToMemory", False)
        _escalation_logger.warning(
            "Failed to ingest escalation outcome into memory space '%s'",
            memory_space_id,
            exc_info=True,
        )
