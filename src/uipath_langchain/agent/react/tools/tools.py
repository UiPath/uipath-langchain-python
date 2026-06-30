"""Control flow tools for agent execution."""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from uipath.agent.react import (
    END_EXECUTION_TOOL,
    RAISE_ERROR_TOOL,
)

from ..utils import (
    build_conversational_end_execution_input_schema,
    has_custom_conversational_output_fields,
)


def create_end_execution_tool(
    agent_output_schema: type[BaseModel] | None = None,
    *,
    is_conversational: bool = False,
) -> StructuredTool:
    """Never executed - routing intercepts and extracts args for successful termination.

    For conversational agents the argument schema is the user's output schema with the
    implicit `uipath__agent_response_messages` field removed — only the custom fields
    are surfaced to the LLM.
    """
    input_schema = agent_output_schema or END_EXECUTION_TOOL.args_schema
    if is_conversational:
        input_schema = build_conversational_end_execution_input_schema(input_schema)

    async def end_execution_fn(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    return StructuredTool(
        name=END_EXECUTION_TOOL.name,
        description=END_EXECUTION_TOOL.description,
        args_schema=input_schema,
        coroutine=end_execution_fn,
    )


def create_raise_error_tool() -> StructuredTool:
    """Never executed - routing intercepts and raises AgentRuntimeError."""

    async def raise_error_fn(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    return StructuredTool(
        name=RAISE_ERROR_TOOL.name,
        description=RAISE_ERROR_TOOL.description,
        args_schema=RAISE_ERROR_TOOL.args_schema,
        coroutine=raise_error_fn,
    )


def create_flow_control_tools(
    agent_output_schema: type[BaseModel] | None = None,
    *,
    is_conversational: bool = False,
) -> list[BaseTool]:
    """Build the flow-control tools the LLM may call to finalize execution.

    Autonomous: `[end_execution, raise_error]` regardless of schema.
    Conversational + custom output schema: `[end_execution]` only.
        Raising errors doesn't apply since conversational agents always respond.
    Conversational + no custom output schema: `[]` — The agent simply terminates by
        emitting an AIMessage without tool calls.
    """
    if is_conversational:
        if has_custom_conversational_output_fields(agent_output_schema):
            return [
                create_end_execution_tool(agent_output_schema, is_conversational=True)
            ]
        return []

    return [
        create_end_execution_tool(agent_output_schema),
        create_raise_error_tool(),
    ]
