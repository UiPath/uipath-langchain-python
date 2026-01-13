from typing import Any

from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel

from uipath_langchain.agent.react.job_attachments import (
    get_job_attachment_paths,
    get_job_attachments,
    replace_job_attachment_ids,
)
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import AsyncToolWrapperType


def get_job_attachment_wrapper(output_type: Any | None = None) -> AsyncToolWrapperType:
    """Create a tool wrapper that handles job attachments in both tool inputs and outputs.

    This wrapper performs two main functions:
    1. Input Processing: Extracts job attachment paths from the tool's schema, validates that
       all referenced attachments exist in the agent state, and replaces attachment IDs with
       complete attachment objects before invoking the tool.
    2. Output Processing: If output_type is provided, extracts job attachments from the tool's
       output and adds them to the agent's inner_state for use in subsequent tool calls.

    Args:
        output_type: Optional Pydantic model type that defines the structure of the tool's output.
                    If provided, the wrapper will extract job attachments from the output.

    Returns:
        An async tool wrapper function that handles job attachment validation, replacement,
        and extraction from tool outputs
    """

    async def job_attachment_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> dict[str, Any] | Command[Any] | None:
        """Validate and replace job attachments in tool arguments, invoke tool, and extract output attachments.

        Processing flow:
        1. Validates and replaces job attachment IDs in tool input arguments with full attachment objects
        2. Invokes the tool with the modified arguments
        3. Extracts job attachments from tool output (if output_type was provided to the wrapper)
        4. Returns a Command object containing the tool result message and updated inner_state with extracted attachments

        Args:
            tool: The tool to wrap
            call: The tool call containing arguments
            state: The agent graph state containing job attachments

        Returns:
            Command object with tool result message and updated job attachments in inner_state,
            or error dict if attachment validation fails
        """
        input_args = call["args"]
        modified_input_args = input_args

        if isinstance(tool.args_schema, type) and issubclass(
            tool.args_schema, BaseModel
        ):
            errors: list[str] = []
            paths = get_job_attachment_paths(tool.args_schema)
            modified_input_args = replace_job_attachment_ids(
                paths, input_args, state.inner_state.job_attachments, errors
            )

            if errors:
                return {"error": "\n".join(errors)}

        tool_result = await tool.ainvoke(modified_input_args)
        job_attachments_dict = {}
        if output_type is not None:
            job_attachments = get_job_attachments(output_type, tool_result)
            job_attachments_dict = {
                str(att.id): att for att in job_attachments if att.id is not None
            }

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=str(tool_result),
                        name=call["name"],
                        tool_call_id=call["id"],
                    )
                ],
                "inner_state": {"job_attachments": job_attachments_dict},
            }
        )

    return job_attachment_wrapper
