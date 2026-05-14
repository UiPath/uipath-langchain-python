"""Factory for creating client-side tools that execute on the client SDK."""

import json
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from uipath.agent.models.agent import AgentClientSideToolResourceConfig
from uipath.eval.mocks import mockable

from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model as create_model_from_schema,
)
from uipath_langchain.chat.hitl import CLIENT_SIDE_TOOL_MARKER

from .utils import sanitize_tool_name


def create_client_side_tool(
    resource: AgentClientSideToolResourceConfig,
) -> StructuredTool:
    """Create a client-side tool that pauses the graph and waits for the client to execute it.

    The tool uses @durable_interrupt to suspend the graph. The client SDK receives
    an executingToolCall event, runs its registered handler, and sends endToolCall
    back through CAS. The bridge routes that endToolCall to wait_for_resume(),
    which unblocks the graph with the client's result.
    """
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model_from_schema(resource.input_schema)

    async def client_side_tool_fn(
        *, tool_call_id: Annotated[str, InjectedToolCallId], **kwargs: Any
    ) -> Any:
        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=(resource.output_schema or {}),
            example_calls=getattr(resource.properties, "example_calls", None),
        )
        async def execute_tool() -> dict[str, Any]:
            """Execute client-side tool, pausing for client response."""

            @durable_interrupt
            async def wait_for_client_execution() -> dict[str, Any]:
                return {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "input": kwargs,
                    "is_execution_phase": True,
                }

            result = await wait_for_client_execution()
            return result.get("output", result) if isinstance(result, dict) else result

        result = await execute_tool()

        if isinstance(result, dict):
            try:
                content = json.dumps(result)
            except TypeError:
                content = str(result)
        else:
            content = str(result) if result is not None else ""

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            response_metadata={CLIENT_SIDE_TOOL_MARKER: True},
        )

    tool = StructuredTool(
        name=tool_name,
        description=resource.description or f"Client-side tool: {tool_name}",
        args_schema=input_model,
        coroutine=client_side_tool_fn,
        metadata={
            CLIENT_SIDE_TOOL_MARKER: True,
            "output_schema": resource.output_schema,
        },
    )

    return tool
