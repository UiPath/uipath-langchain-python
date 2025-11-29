"""Escalation tool creation for Action Center integration."""

from __future__ import annotations

import os
import uuid
from typing import Any

from jsonschema_pydantic_converter import transform as create_model
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import TypeAdapter
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateEscalation

from .utils import sanitize_tool_name


def create_escalation_tool(resource: AgentEscalationResourceConfig) -> StructuredTool:
    """Uses interrupt() for Action Center human-in-the-loop. Returns Command with response in state."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)

    # TODO: only works for UserEmail type as value is GUID for UserId or GroupId for other types but the API expects strings e.g: email@domain.com
    # we need to do user resolution via Identity here
    assignee: str | None = (
        channel.recipients[0].value
        if channel.recipients
        and channel.recipients[0].type == AgentEscalationRecipientType.USER_EMAIL
        else None
    )

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def escalation_tool_fn(**kwargs: Any) -> Any:
        # Get execution context from environment variables
        instance_id = os.getenv("UIPATH_TRACE_ID") or str(uuid.uuid4())
        job_key = os.getenv("UIPATH_JOB_KEY")
        process_key = os.getenv("UIPATH_PROCESS_KEY")

        result = interrupt(
            CreateEscalation(
                title=channel.task_title or resource.name,
                data=kwargs,
                assignee=assignee,
                app_name=channel.properties.app_name,
                app_folder_path=channel.properties.folder_name,
                app_key=channel.properties.app_name,
                app_version=channel.properties.app_version,
                priority=channel.priority,
                labels=channel.labels,
                is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
                actionable_message_metadata=channel.properties.actionable_message_meta_data,
                agent_id=resource.id,
                resource_key=channel.properties.resource_key,
            )
        )

        return TypeAdapter(output_model).validate_python(result)

    escalation_tool_fn.__annotations__["return"] = output_model

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=escalation_tool_fn,
    )

    tool.__dict__["OutputType"] = output_model

    return tool
