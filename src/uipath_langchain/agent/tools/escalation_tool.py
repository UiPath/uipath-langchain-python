"""Escalation tool creation for Action Center integration."""

from __future__ import annotations

import logging
import os
import uuid
from enum import Enum
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from uipath._utils.constants import ENV_JOB_KEY
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateEscalation
from uipath.utils.dynamic_schema import jsonschema_to_pydantic

from ..react.types import AgentGraphNode, AgentTerminationSource
from .utils import sanitize_tool_name

logger = logging.getLogger(__name__)


class EscalationAction(str, Enum):
    """Actions that can be taken after an escalation completes."""

    CONTINUE = "continue"
    END = "end"


def create_escalation_tool(resource: AgentEscalationResourceConfig) -> StructuredTool:
    """Uses interrupt() for Action Center human-in-the-loop. Returns Command with response in state."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: type[BaseModel] = jsonschema_to_pydantic(channel.input_schema)
    output_model: type[BaseModel] = jsonschema_to_pydantic(channel.output_schema)

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
    async def escalation_tool_fn(**kwargs: Any) -> Command | dict:
        instance_id = os.getenv("UIPATH_TRACE_ID") or str(uuid.uuid4())
        job_key = os.getenv(ENV_JOB_KEY)
        process_key = os.getenv("UIPATH_PROCESS_KEY")
        task_title = channel.task_title or "Escalation Task"

        result = interrupt(
            CreateEscalation(
                title=task_title,
                data=kwargs,
                assignee=assignee,
                app_name=channel.properties.app_name,
                app_folder_path=channel.properties.folder_name,
                app_version=channel.properties.app_version,
                priority=channel.priority,
                labels=channel.labels,
                is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
                actionable_message_metadata=channel.properties.actionable_message_meta_data,
                agent_id=resource.id,
                instance_id=instance_id,
                job_key=job_key,
                process_key=process_key,
                resource_key=channel.properties.resource_key,
            )
        )

        escalation_action = getattr(result, "action", None)
        escalation_output = getattr(result, "data", {})
        validated_result = output_model.model_validate(escalation_output)

        outcome = (
            channel.outcome_mapping.get(escalation_action)
            if channel.outcome_mapping and escalation_action
            else None
        )
        if outcome and outcome == EscalationAction.END:
            return Command(
                update={
                    "termination": {
                        "source": AgentTerminationSource.ESCALATION,
                        "title": f"Agent run ended based on escalation outcome {outcome} with directive {escalation_action}",
                        "detail": f"Escalation output: {validated_result.model_dump()}",
                    }
                },
                goto=AgentGraphNode.TERMINATE,
            )

        return validated_result.model_dump()

    escalation_tool_fn.__annotations__["return"] = output_model

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=escalation_tool_fn,
    )

    tool.__dict__["OutputType"] = output_model

    return tool
