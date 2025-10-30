"""Escalation tool creation for Action Center integration."""

from __future__ import annotations

from typing import Any, Type

from jsonschema_pydantic import jsonschema_to_pydantic  # type: ignore[import-untyped]
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentEscalationChannel,
    # AgentEscalationRecipientType,
    AgentEscalationResourceConfig,
)
from uipath.models import CreateAction

from .utils import sanitize_tool_name


def create_escalation_tool(resource: AgentEscalationResourceConfig) -> StructuredTool:
    """Uses interrupt() for Action Center human-in-the-loop. Returns Command with response in state."""

    tool_name: str = f"escalate_{sanitize_tool_name(resource.name)}"
    channel: AgentEscalationChannel = resource.channels[0]

    input_model: Type[BaseModel] = jsonschema_to_pydantic(channel.input_schema)
    output_model: Type[BaseModel] = jsonschema_to_pydantic(channel.output_schema)

    # TODO: only works for UserEmail type as value is GUID for UserId or GroupId for other types but the API expects strings e.g: email@domain.com
    # we need to do user resolution via Identity here
    assignee: str | None = None
    #     (
    #     channel.recipients[0].value
    #     if channel.recipients
    #     and channel.recipients[0].type == AgentEscalationRecipientType.USER_EMAIL
    #     else None
    # )

    async def escalation_tool_fn(**kwargs: Any):
        try:
            result = interrupt(
                CreateAction(
                    title=channel.task_title,
                    data=kwargs,
                    assignee=assignee,
                    app_name=channel.properties.app_name,
                    app_folder_path=channel.properties.folder_name,
                    app_key=channel.properties.app_name,
                    app_version=channel.properties.app_version,
                )
            )
        except Exception:
            raise

        return result

    class EscalationTool(StructuredTool):
        """Escalation tool with OutputType for schema compatibility."""

        OutputType: Type[BaseModel] = output_model

    return EscalationTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=escalation_tool_fn,
    )
