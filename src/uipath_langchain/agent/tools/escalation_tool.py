"""Escalation tool creation for Action Center integration."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.types import Command, interrupt
from pydantic import BaseModel, create_model
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
    """Uses interrupt() for Action Center human-in-the-loop."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]
    base_input_model: type[BaseModel] = jsonschema_to_pydantic(channel.input_schema)
    input_model = create_model(
        base_input_model.__name__,
        __base__=base_input_model,
        tool_call_id=(Annotated[str, InjectedToolCallId]),
    )
    output_model: type[BaseModel] = jsonschema_to_pydantic(channel.output_schema)

    # only works for UserEmail type as value is GUID for UserId or GroupId for other types
    # but the API expects strings e.g: email@domain.com
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
    async def escalation_tool_fn(
        tool_call_id: Annotated[str, InjectedToolCallId], **kwargs: Any
    ) -> Command[Any] | Any:
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
                    "messages": [
                        ToolMessage(
                            content="Terminating the agent execution as configured in the escalation outcome",
                            tool_call_id=tool_call_id,
                        )
                    ],
                    "termination": {
                        "source": AgentTerminationSource.ESCALATION,
                        "title": f"Agent run ended based on escalation outcome {outcome} with directive {escalation_action}",
                        "detail": f"Escalation output: {validated_result.model_dump()}",
                    },
                },
                goto=AgentGraphNode.TERMINATE,
            )

        return validated_result

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=escalation_tool_fn,
    )

    return tool
