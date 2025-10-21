"""Control flow tools for agent execution."""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from uipath.agent.react import (
    END_EXECUTION_TOOL,
    RAISE_ERROR_TOOL,
)


def create_end_execution_tool(
    agent_output_schema: type[BaseModel] | None = None,
) -> StructuredTool:
    """Never executed - routing intercepts and extracts args for successful termination."""
    input_schema = agent_output_schema or END_EXECUTION_TOOL.args_schema

    async def end_execution_fn(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    return StructuredTool(
        name=END_EXECUTION_TOOL.name,
        description=END_EXECUTION_TOOL.description,
        args_schema=input_schema,
        coroutine=end_execution_fn,
    )


def create_raise_error_tool() -> StructuredTool:
    """Never executed - routing intercepts and raises AgentTerminationException."""

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
) -> list[BaseTool]:
    return [
        create_end_execution_tool(agent_output_schema),
        create_raise_error_tool(),
    ]
