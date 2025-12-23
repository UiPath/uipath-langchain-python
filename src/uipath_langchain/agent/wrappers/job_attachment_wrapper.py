from typing import Any

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel

from uipath_langchain.agent.react.job_attachments import (
    get_job_attachment_paths,
    replace_job_attachment_ids,
)
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import AsyncToolWrapperType


def get_job_attachment_wrapper() -> AsyncToolWrapperType:
    """Create a tool wrapper that validates and replaces job attachment IDs with full attachment objects.

    This wrapper extracts job attachment paths from the tool's schema, validates that all
    referenced attachments exist in the agent state, and replaces attachment IDs with complete
    attachment objects before invoking the tool.

    Args:
        resource: The agent tool resource configuration

    Returns:
        An async tool wrapper function that handles job attachment validation and replacement
    """

    async def job_attachment_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> dict[str, Any] | Command[Any] | None:
        """Validate and replace job attachments in tool arguments before invocation.

        Args:
            tool: The tool to wrap
            call: The tool call containing arguments
            state: The agent graph state containing job attachments

        Returns:
            Tool invocation result, or error dict if attachment validation fails
        """
        input_args = call["args"]
        modified_input_args = input_args

        if isinstance(tool.args_schema, type) and issubclass(
            tool.args_schema, BaseModel
        ):
            errors: list[str] = []
            paths = get_job_attachment_paths(tool.args_schema)
            modified_input_args = replace_job_attachment_ids(
                paths, input_args, state.job_attachments, errors
            )

            if errors:
                return {"error": "\n".join(errors)}

        return await tool.ainvoke(modified_input_args)

    return job_attachment_wrapper
