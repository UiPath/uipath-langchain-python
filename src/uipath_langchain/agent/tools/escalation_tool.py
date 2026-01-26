"""Escalation tool creation for Action Center integration."""

from enum import Enum
from typing import Any

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    AssetRecipient,
    StandardRecipient,
    TextBuilderTaskTitle,
)
from uipath.agent.utils.text_tokens import build_string_from_tokens
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath.platform.common import CreateEscalation
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.static_args import (
    handle_static_args,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..exceptions import AgentTerminationException
from .tool_node import ToolWrapperReturnType
from ..react.types import AgentGraphState
from .utils import sanitize_dict_for_serialization, sanitize_tool_name


class EscalationAction(str, Enum):
    """Actions that can be taken after an escalation completes."""

    CONTINUE = "continue"
    END = "end"


async def resolve_recipient_value(
    recipient: AgentEscalationRecipient,
) -> TaskRecipient | None:
    """Resolve recipient value based on recipient type."""
    if isinstance(recipient, AssetRecipient):
        value = await resolve_asset(recipient.asset_name, recipient.folder_path)
        type = None
        if recipient.type == AgentEscalationRecipientType.ASSET_USER_EMAIL:
            type = TaskRecipientType.EMAIL
        elif recipient.type == AgentEscalationRecipientType.ASSET_GROUP_NAME:
            type = TaskRecipientType.GROUP_NAME
        return TaskRecipient(value=value, type=type)

    if isinstance(recipient, StandardRecipient):
        type = TaskRecipientType(recipient.type)
        if recipient.type == AgentEscalationRecipientType.USER_EMAIL:
            type = TaskRecipientType.EMAIL
        return TaskRecipient(value=recipient.value, type=type)

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


def create_escalation_tool(
    resource: AgentEscalationResourceConfig,
) -> StructuredTool:
    """Uses interrupt() for Action Center human-in-the-loop."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Any = create_model(channel.input_schema)
    output_model: Any = create_model(channel.output_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=channel.properties.example_calls,
    )
    async def escalation_tool_fn(**kwargs: Any) -> dict[str, Any]:
        # Get task title: use built string from tokens if TEXT_BUILDER, otherwise use directly
        if isinstance(channel.task_title, TextBuilderTaskTitle):
            if tool.metadata is None:
                raise RuntimeError(
                    "Tool metadata is required for TEXT_BUILDER task titles"
                )
            task_title = tool.metadata["taskTitle"]
        elif isinstance(channel.task_title, str):
            task_title = channel.task_title
        else:
            raise NotImplementedError(
                f"TaskTitle type '{type(channel.task_title).__name__}' not implemented"
            )

        recipient: TaskRecipient | None = (
            await resolve_recipient_value(channel.recipients[0])
            if channel.recipients
            else None
        )

        # Recipient requires runtime resolution, store in metadata after resolving
        if tool.metadata is not None:
            tool.metadata["recipient"] = recipient

        result = interrupt(
            CreateEscalation(
                title=task_title,
                data=kwargs,
                recipient=recipient,
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
    ) -> ToolWrapperReturnType:
        # Build task title from tokens if it's a TEXT_BUILDER type
        if isinstance(channel.task_title, TextBuilderTaskTitle):
            if tool.metadata is None:
                raise RuntimeError(
                    "Tool metadata is required for TEXT_BUILDER task titles"
                )
            tool.metadata["taskTitle"] = build_string_from_tokens(
                channel.task_title.tokens, sanitize_dict_for_serialization(dict(state))
            )
        call["args"] = handle_static_args(resource, state, call["args"])
        result = await tool.ainvoke(call["args"])

        if result["action"] == EscalationAction.END:
            output_detail = f"Escalation output: {result['output']}"
            termination_title = (
                f"Agent run ended based on escalation outcome {result['action']} "
                f"with directive {result['escalation_action']}"
            )

            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title=termination_title,
                detail=output_detail,
            )

        return result["output"]

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
        },
    )
    tool.set_tool_wrappers(awrapper=escalation_wrapper)

    return tool
