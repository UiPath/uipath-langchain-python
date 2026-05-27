"""Factory for creating client-side tools that execute on the client SDK."""

import json
from contextvars import ContextVar
from typing import Annotated, Any, TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from uipath.agent.models.agent import AgentClientSideToolResourceConfig
from uipath.eval.mocks import mockable

from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model as create_model_from_schema,
)
from uipath_langchain.chat.hitl import IS_CONVERSATIONAL_CLIENT_SIDE_TOOL

from .utils import sanitize_tool_name

# When set, only tools in this set are available for the current exchange.
# None means all client-side tools are available (default for CAS/web UI).
available_client_side_tools: ContextVar[set[str] | None] = ContextVar(
    "available_client_side_tools", default=None
)

UIPATH_CLIENT_SIDE_TOOLS_INPUT_KEY = "uipath__client_side_tools"


class ClientSideToolInfo(TypedDict):
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None


def apply_tool_filter(
    declared_tools: list[str | dict[str, Any]],
    agent_tools: dict[str, ClientSideToolInfo],
) -> None:
    """Filter available client-side tools to the intersection of declared and agent tools.

    Extracts tool names from the client's declarations, intersects with the agent's
    defined client-side tools, and sets the availability filter. Unknown names are
    silently ignored.

    Args:
        declared_tools: List of tool names (strings) or dicts with a 'name' field
            from uipath__client_side_tools input.
        agent_tools: The agent's client-side tools keyed by name.
    """
    declared_names: set[str] = set()
    for t in declared_tools:
        if isinstance(t, str):
            declared_names.add(t)
        elif isinstance(t, dict) and "name" in t:
            declared_names.add(t["name"])

    available_client_side_tools.set(declared_names & set(agent_tools.keys()))


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
        allowed = available_client_side_tools.get()
        if allowed is not None and tool_name not in allowed:
            return ToolMessage(
                content=f"Tool '{tool_name}' is not available — the client has not registered a handler for it.",
                tool_call_id=tool_call_id,
                status="error",
            )

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
            response_metadata={IS_CONVERSATIONAL_CLIENT_SIDE_TOOL: True},
        )

    tool = StructuredTool(
        name=tool_name,
        description=resource.description or f"Client-side tool: {tool_name}",
        args_schema=input_model,
        coroutine=client_side_tool_fn,
        metadata={
            IS_CONVERSATIONAL_CLIENT_SIDE_TOOL: True,
            "output_schema": resource.output_schema,
        },
    )

    return tool
