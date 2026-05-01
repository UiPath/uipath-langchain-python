"""Factory for creating client-side tools that execute on the client SDK."""

import inspect
import json
from logging import getLogger
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from uipath.agent.models.agent import AgentClientSideToolResourceConfig
from uipath.eval.mocks import mockable

from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model as create_model_from_schema,
)

from .utils import sanitize_tool_name

logger = getLogger(__name__)

CLIENT_SIDE_TOOL_MARKER = "uipath_client_tool"


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
            example_calls=getattr(resource.properties, 'example_calls', None),
        )
        @durable_interrupt
        async def wait_for_client_execution() -> dict[str, Any]:
            return {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "input": kwargs,
                "is_execution_phase": True,
            }

        # First run: raises GraphInterrupt with the tool call info.
        # On resume: returns the client's result (output, isError, etc.)
        # During evals: @mockable intercepts and returns simulated response.
        result = await wait_for_client_execution()

        # The resume value from the bridge is the endToolCall payload
        output = result.get("output")
        is_error = result.get("is_error", False)

        content = str(output) if output is not None else ""
        if isinstance(output, dict):
            content = json.dumps(output)

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            status="error" if is_error else "success",
            response_metadata={CLIENT_SIDE_TOOL_MARKER: True},
        )

    # Patch signature so LangChain injects tool_call_id at runtime
    original_sig = inspect.signature(client_side_tool_fn)
    params = [p for p in original_sig.parameters.values() if p.name != "kwargs"] + [
        inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD, annotation=Any),
    ]
    client_side_tool_fn.__signature__ = original_sig.replace(parameters=params)

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
