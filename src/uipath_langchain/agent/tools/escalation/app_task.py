"""Action Center *app* escalation (``escalationType=0``).

Creates an Action Center task against an app, suspends execution via
``durable_interrupt`` until the task is completed, and optionally
records the outcome to escalation memory.
"""

import json
import logging
import os
from typing import Any

from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationResourceConfig,
    LowCodeAgentDefinition,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import Task
from uipath.platform.common import WaitEscalation

from uipath_langchain._utils import get_current_span_and_trace_ids
from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..utils import sanitize_tool_name
from .common import (
    _resolve_escalation_action,
    build_invocation_ctx,
    finalize_escalation_result,
    make_escalation_tool_output,
    make_escalation_wrapper,
)
from .memory import (
    EscalationMemorySettings,
    _check_escalation_memory_cache,
    _get_escalation_memory_folder_path,
    _get_escalation_memory_settings,
    _get_escalation_memory_space_id,
    _get_escalation_memory_space_name,
    _ingest_escalation_memory,
    _resolve_user_id,
)

_escalation_logger = logging.getLogger(__name__)


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


def create_escalation_tool(
    resource: AgentEscalationResourceConfig,
    agent: LowCodeAgentDefinition | None = None,
) -> StructuredTool:
    """Action Center app-task escalation (``escalationType=0``).

    Uses ``durable_interrupt`` for Action Center human-in-the-loop and
    optionally writes the outcome to escalation memory.
    """

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)
    EscalationToolOutput = make_escalation_tool_output(output_model)

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
        ctx = await build_invocation_ctx(tool, channel, kwargs, input_model)

        # --- Escalation memory: check cache before creating HITL task ---
        if _memory_space_id:
            cached_result = await _check_escalation_memory_cache(
                _memory_space_id,
                ctx.serialized_data,
                folder_path=_memory_folder_path or ctx.folder_path,
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
                created_task = await client.tasks.create_async(
                    title=ctx.task_title,
                    data=ctx.serialized_data,
                    app_name=channel.properties.app_name,
                    app_folder_path=ctx.folder_path,
                    recipient=ctx.recipient,
                    priority=channel.priority,
                    labels=channel.labels,
                    is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
                    actionable_message_metadata=channel.properties.actionable_message_meta_data,
                )

                if created_task.id is not None:
                    _bts_context["task_key"] = str(created_task.id)

                return WaitEscalation(
                    action=created_task,
                    app_folder_path=ctx.folder_path,
                    app_name=channel.properties.app_name,
                    recipient=ctx.recipient,
                )

            return await create_escalation_task()

        result = await escalate(**kwargs)
        if isinstance(result, dict):
            result = Task.model_validate(result)

        finalized = finalize_escalation_result(
            result,
            input_model=input_model,
            output_model=output_model,
            outcome_mapping=channel.outcome_mapping,
        )

        # --- Escalation memory: persist outcome for future recall ---
        if _memory_space_id and not result.is_deleted:
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
                return finalized

            answer_payload, attributes_payload = _build_escalation_memory_payload(
                ctx.serialized_data,
                finalized["output"],
                finalized["outcome"],
            )
            await _ingest_escalation_memory(
                _memory_space_id,
                answer=json.dumps(answer_payload),
                attributes=json.dumps(attributes_payload),
                parent_span_id=parent_span_id,
                trace_id=trace_id,
                user_id=user_id,
                folder_path=_memory_folder_path or ctx.folder_path,
            )
            if user_id is None:
                _escalation_logger.info(
                    "Ingested escalation memory without reviewer user ID "
                    "because the completed user could not be resolved"
                )

        return finalized

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
            "_span_context": _span_context,
            "_bts_context": _bts_context,
        },
    )
    tool.set_tool_wrappers(awrapper=make_escalation_wrapper(channel))

    return tool
