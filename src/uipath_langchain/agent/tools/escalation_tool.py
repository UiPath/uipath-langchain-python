"""Escalation tool creation for Action Center integration."""

from enum import Enum
from typing import Any

from jsonschema_pydantic_converter import transform as create_model
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command, interrupt
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import CreateEscalation

from .utils import sanitize_tool_name


class EscalationAction(str, Enum):
    """Actions that can be taken after an escalation completes."""

    CONTINUE = "continue"
    END = "end"


def resolve_recipient_value(recipient: AgentEscalationRecipient) -> str | None:
    """Resolve recipient value based on recipient type."""
    if recipient.type == AgentEscalationRecipientType.ASSET_USER_EMAIL:
        return resolve_asset(recipient.asset_name, recipient.folder_path)

    # For USER_EMAIL, USER_ID, and GROUP_ID, return the value directly
    if hasattr(recipient, "value"):
        return recipient.value

    return None


def resolve_asset(asset_name: str, folder_path: str) -> str | None:
    """Retrieve asset value."""
    try:
        client = UiPath()
        result = client.assets.retrieve(name=asset_name, folder_path=folder_path)

        if not result or not result.value:
            raise ValueError(f"Asset '{asset_name}' has no value configured.")

        return result.value
    except Exception as e:
        raise ValueError(
            f"Failed to resolve asset '{asset_name}' in folder '{folder_path}': {str(e)}"
        ) from e


def create_escalation_tool(resource: AgentEscalationResourceConfig) -> StructuredTool:
    """Uses interrupt() for Action Center human-in-the-loop."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)

    assignee: str | None = (
        resolve_recipient_value(channel.recipients[0]) if channel.recipients else None
    )

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def escalation_tool_fn(
        runtime: ToolRuntime, **kwargs: Any
    ) -> Command[Any] | Any:
        from ..react.types import AgentGraphNode, AgentTerminationSource

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

        outcome = (
            channel.outcome_mapping.get(escalation_action)
            if channel.outcome_mapping and escalation_action
            else None
        )

        if outcome == EscalationAction.END:
            output_detail = f"Escalation output: {escalation_output}"
            termination_title = f"Agent run ended based on escalation outcome {outcome} with directive {escalation_action}"

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"{termination_title}. {output_detail}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                    "termination": {
                        "source": AgentTerminationSource.ESCALATION,
                        "title": termination_title,
                        "detail": output_detail,
                    },
                },
                goto=AgentGraphNode.TERMINATE,
            )

        return escalation_output

    tool = StructuredTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=escalation_tool_fn,
    )

    return tool
