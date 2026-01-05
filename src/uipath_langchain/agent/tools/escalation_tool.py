"""Escalation tool creation for Action Center integration."""

from enum import Enum
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command, interrupt
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipient,
    AgentEscalationResourceConfig,
    AssetRecipient,
    StandardRecipient,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import CreateEscalation

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model

from ..react.types import AgentGraphNode, AgentGraphState, AgentTerminationSource
from .tool_node import ToolWrapperMixin
from .utils import sanitize_tool_name


class EscalationAction(str, Enum):
    """Actions that can be taken after an escalation completes."""

    CONTINUE = "continue"
    END = "end"


async def resolve_recipient_value(recipient: AgentEscalationRecipient) -> str | None:
    """Resolve recipient value based on recipient type."""
    if isinstance(recipient, AssetRecipient):
        return await resolve_asset(recipient.asset_name, recipient.folder_path)

    if isinstance(recipient, StandardRecipient):
        return recipient.value

    return None


async def resolve_asset(asset_name: str, folder_path: str) -> str | None:
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


class StructuredToolWithWrapper(StructuredTool, ToolWrapperMixin):
    pass


async def create_escalation_tool(
    resource: AgentEscalationResourceConfig,
) -> StructuredTool:
    """Uses interrupt() for Action Center human-in-the-loop."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)

    assignee: str | None = (
        await resolve_recipient_value(channel.recipients[0])
        if channel.recipients
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

        return {
            "action": outcome,
            "output": escalation_output,
            "escalation_action": escalation_action,
        }

    async def escalation_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> dict[str, Any] | Command[Any]:
        result = await tool.ainvoke(call["args"])

        if result["action"] == EscalationAction.END:
            output_detail = f"Escalation output: {result['output']}"
            termination_title = (
                f"Agent run ended based on escalation outcome {result['action']} "
                f"with directive {result['escalation_action']}"
            )

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"{termination_title}. {output_detail}",
                            tool_call_id=call["id"],
                        )
                    ],
                    "inner_state": {
                        "termination": {
                            "source": AgentTerminationSource.ESCALATION,
                            "title": termination_title,
                            "detail": output_detail,
                        }
                    },
                },
                goto=AgentGraphNode.TERMINATE,
            )

        return result["output"]

    tool = StructuredToolWithWrapper(
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
    tool.set_tool_wrappers(awrapper=escalation_wrapper)

    return tool
