"""Escalation tool creation for Action Center integration."""

from enum import Enum
from typing import Any, Literal

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    AgentEscalationRecipient,
    AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
    AssetRecipient,
    StandardRecipient,
    TaskTitle,
    TextBuilderTaskTitle,
)
from uipath.agent.utils.text_tokens import build_string_from_tokens
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import TaskRecipient, TaskRecipientType
from uipath.platform.common import CreateEscalation, UiPathConfig
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.static_args import (
    handle_static_args,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from ..exceptions import AgentTerminationException
from ..react.types import AgentGraphState
from .tool_node import ToolWrapperReturnType
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


def _resolve_task_title(
    task_title: TaskTitle | str | None, agent_input: dict[str, Any]
) -> str:
    """Resolve task title based on channel configuration."""
    if isinstance(task_title, TextBuilderTaskTitle):
        return build_string_from_tokens(task_title.tokens, agent_input)

    if isinstance(task_title, str):
        return task_title

    return "Escalation Task"


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

    async def escalation_tool_fn(**kwargs: Any) -> dict[str, Any]:
        recipient: TaskRecipient | None = (
            await resolve_recipient_value(channel.recipients[0])
            if channel.recipients
            else None
        )

        task_title = "Escalation Task"
        if tool.metadata is not None:
            # Recipient requires runtime resolution, store in metadata after resolving
            tool.metadata["recipient"] = recipient
            task_title = tool.metadata.get("task_title") or task_title

        @mockable(
            name=tool_name.lower(),
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=EscalationToolOutput.model_json_schema(),
            example_calls=channel.properties.example_calls,
        )
        async def escalate():
            return interrupt(
                CreateEscalation(
                    title=task_title,
                    data=kwargs,
                    recipient=recipient,
                    app_name=channel.properties.app_name,
                    app_folder_path=channel.properties.folder_name,
                    priority=channel.priority,
                    labels=channel.labels,
                    is_actionable_message_enabled=channel.properties.is_actionable_message_enabled,
                    actionable_message_metadata=channel.properties.actionable_message_meta_data,
                )
            )

        result = await escalate()

        # Extract task info before validation
        task_id = result.id
        task_url = f"{UiPathConfig.base_url}/actions_/tasks/{task_id}"
        assigned_to = _get_user_email(result.assigned_to_user)

        outcome = result.action
        escalation_output = _parse_task_data(
            result.data,
            input_schema=input_model.model_json_schema(),
            output_schema=EscalationToolOutput.model_json_schema(),
        )

        outcome_str = (
            channel.outcome_mapping.get(outcome)
            if channel.outcome_mapping and outcome
            else None
        )
        escalation_action = (
            EscalationAction(outcome_str) if outcome_str else EscalationAction.CONTINUE
        )

        return {
            "action": escalation_action,
            "output": escalation_output,
            "outcome": outcome,
            "task_id": task_id,
            "task_url": task_url,
            "assigned_to": assigned_to,
        }

    async def escalation_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        if tool.metadata is None:
            raise RuntimeError("Tool metadata is required for task_title resolution")

        tool.metadata["task_title"] = _resolve_task_title(
            channel.task_title, sanitize_dict_for_serialization(dict(state))
        )

        tool.metadata["_call_id"] = call.get("id")
        tool.metadata["_call_args"] = dict(call.get("args", {}))

        call["args"] = handle_static_args(resource, state, call["args"])
        result = await tool.ainvoke(call["args"])

        if result["action"] == EscalationAction.END:
            output_detail = f"Escalation output: {result['output']}"
            termination_title = (
                f"Agent run ended based on escalation outcome {result['action']} "
                f"with directive {result['outcome']}"
            )

            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title=termination_title,
                detail=output_detail,
            )

        return {
            "output": result["output"],
            "outcome": result["outcome"],
            "task_id": result["task_id"],
            "assigned_to": result["assigned_to"],
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
        },
    )
    tool.set_tool_wrappers(awrapper=escalation_wrapper)

    return tool
