"""Escalation tool creation for Action Center integration."""

from enum import Enum
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform.common import CreateEscalation

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model

from ..exceptions import AgentToolTerminationRequest
from ..react.types import AgentTerminationSource
from .utils import sanitize_tool_name


class EscalationAction(str, Enum):
    """Actions that can be taken after an escalation completes."""

    CONTINUE = "continue"
    END = "end"


def create_escalation_tool(resource: AgentEscalationResourceConfig) -> BaseTool:
    """Uses interrupt() for Action Center human-in-the-loop."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)

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
        example_calls=channel.properties.example_calls,
    )
    async def escalation_tool_fn(**kwargs: Any) -> dict[str, Any]:
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

        outcome_str = (
            channel.outcome_mapping.get(escalation_action)
            if channel.outcome_mapping and escalation_action
            else None
        )
        outcome = (
            EscalationAction(outcome_str) if outcome_str else EscalationAction.CONTINUE
        )

        if outcome == EscalationAction.END:
            raise AgentToolTerminationRequest(
                source=AgentTerminationSource.ESCALATION,
                title=(
                    f"Agent run ended based on escalation outcome {outcome} "
                    f"with directive {escalation_action}"
                ),
                detail=f"Escalation output: {escalation_output}",
                output=escalation_output,
            )

        return escalation_output

    return StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=escalation_tool_fn,
        metadata={
            "tool_type": "escalation",
            "display_name": channel.properties.app_name,
            "channel_type": channel.type,
            "assignee": assignee,
        },
    )
