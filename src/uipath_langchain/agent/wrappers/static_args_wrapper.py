from typing import Any

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.types import Command
from uipath.agent.models.agent import BaseAgentResourceConfig

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.tool_node import AsyncToolWrapperType


def get_static_args_wrapper(
    resource: BaseAgentResourceConfig,
) -> AsyncToolWrapperType:
    """Returns an asynchronous tool wrapper that applies static arguments.

    Args:
        resource: The agent resource configuration.

    Returns:
        An asynchronous tool wrapper function.
    """

    async def static_args_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> dict[str, Any] | Command[Any] | None:
        input_args = call["args"]
        merged_args = handle_static_args(resource, state, input_args)
        return await tool.ainvoke(merged_args)

    return static_args_wrapper
