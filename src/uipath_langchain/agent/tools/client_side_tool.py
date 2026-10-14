"""Factory for creating client-side tools that execute on the client SDK."""

import json
from contextvars import ContextVar
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from uipath.agent.models.agent import AgentClientSideToolResourceConfig
from uipath.eval.mocks import mockable

from uipath_langchain._utils.durable_interrupt import (
    SUSPENDS_RUN,
    durable_interrupt,
)
from uipath_langchain.agent.contracts.client_side_tools import (
    ClientSideToolInfo as ClientSideToolInfo,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model as create_model_from_schema,
)
from uipath_langchain.chat.hitl import IS_CONVERSATIONAL_CLIENT_SIDE_TOOL

from .utils import sanitize_tool_name

# When set, only tools in this set are available for the current exchange.
# None means all client-side tools are available (default for CAS/web UI).
# In a graph, availability is decided by the tool node from the graph state
# (`declared_client_side_tools`), before any confirmation: LangGraph runs each
# node in a copy of the context, so a value set here in another node (such as
# the init node) is not visible when the tool runs.
available_client_side_tools: ContextVar[set[str] | None] = ContextVar(
    "available_client_side_tools", default=None
)

UIPATH_CLIENT_SIDE_TOOLS_INPUT_KEY = "uipath__client_side_tools"


def declared_tool_names(declared_tools: list[str | dict[str, Any]]) -> set[str]:
    """Tool names from the client's declarations: strings, or dicts with a 'name' field."""
    names: set[str] = set()
    for t in declared_tools:
        if isinstance(t, str):
            names.add(t)
        elif isinstance(t, dict) and "name" in t:
            names.add(t["name"])
    return names


def declared_client_side_tools(state: Any) -> set[str] | None:
    """The client-side tools the exchange declared, read from the graph state.

    None when `uipath__client_side_tools` is absent: every client-side tool is
    available. A list makes only the declared names available, so an empty list
    makes none available.
    """
    declared = getattr(state, UIPATH_CLIENT_SIDE_TOOLS_INPUT_KEY, None)
    return declared_tool_names(declared) if isinstance(declared, list) else None


def client_side_tool_not_available(tool_name: str, tool_call_id: str) -> ToolMessage:
    """The result of calling a client-side tool the client has no handler for."""
    return ToolMessage(
        content=f"Tool '{tool_name}' is not available — the client has not registered a handler for it.",
        tool_call_id=tool_call_id,
        status="error",
    )


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
    declared_names = declared_tool_names(declared_tools)
    available_client_side_tools.set(declared_names & set(agent_tools.keys()))


def create_client_side_tool(
    resource: AgentClientSideToolResourceConfig,
) -> StructuredTool:
    """Create a client-side tool that pauses the graph and waits for the client to execute it.

    The tool uses @durable_interrupt to suspend the graph. The client receives
    an executingToolCall event, executes its registered handler, and sends
    endToolCall back through CAS.
    """
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model_from_schema(resource.input_schema)

    async def client_side_tool_fn(
        *, tool_call_id: Annotated[str, InjectedToolCallId], **kwargs: Any
    ) -> Any:
        allowed = available_client_side_tools.get()
        if allowed is not None and tool_name not in allowed:
            return client_side_tool_not_available(tool_name, tool_call_id)

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
                }

            result = await wait_for_client_execution()
            return result if isinstance(result, dict) else {"output": result}

        result = await execute_tool()

        is_error = result.get("isError", False)
        output = result.get("output", result)

        if isinstance(output, dict):
            try:
                content = json.dumps(output)
            except TypeError:
                content = str(output)
        else:
            content = str(output) if output is not None else ""

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            status="error" if is_error else "success",
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
            SUSPENDS_RUN: True,
        },
    )

    return tool
