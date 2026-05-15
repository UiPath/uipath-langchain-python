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


def validate_and_apply_tool_filter(
    declared_tools: list[dict[str, Any]],
    agent_tools: dict[str, ClientSideToolInfo],
) -> None:
    """Validate client-side tool declarations and set the availability filter.

    Compares the client's declared tools against the agent's tool definitions.
    Raises ValueError if required tools are missing or schemas don't match.
    Sets the available_client_side_tools context variable for tool functions.

    Args:
        declared_tools: List of tool declarations from uipath__client_side_tools input.
            Each item is a dict with 'name' and optional 'inputSchema'/'outputSchema'.
        agent_tools: The agent's client-side tools.
            Dict of {tool_name: ClientSideToolInfo}.
    """
    declared = {
        (t["name"] if isinstance(t, dict) else t): t
        if isinstance(t, dict)
        else {"name": t}
        for t in declared_tools
    }

    required = set(agent_tools.keys())
    missing = required - set(declared.keys())
    if missing:
        raise ValueError(
            f"Missing required client-side tools: {', '.join(sorted(missing))}. "
            f"The client must register handlers for all client-side tools defined by the agent."
        )

    for name, decl in declared.items():
        agent_tool = agent_tools.get(name)
        if agent_tool is None:
            continue  # Unknown tool, runtime will ignore it
        if decl.get("inputSchema") and agent_tool.get("input_schema"):
            if json.dumps(decl["inputSchema"], sort_keys=True) != json.dumps(
                agent_tool["input_schema"], sort_keys=True
            ):
                raise ValueError(
                    f"Client-side tool '{name}' inputSchema does not match agent definition."
                )
        if decl.get("outputSchema") and agent_tool.get("output_schema"):
            if json.dumps(decl["outputSchema"], sort_keys=True) != json.dumps(
                agent_tool["output_schema"], sort_keys=True
            ):
                raise ValueError(
                    f"Client-side tool '{name}' outputSchema does not match agent definition."
                )

    available_client_side_tools.set(set(declared.keys()))


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
