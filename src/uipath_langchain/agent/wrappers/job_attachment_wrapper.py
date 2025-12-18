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


async def job_attachment_wrapper(
    tool: BaseTool,
    call: ToolCall,
    state: AgentGraphState,
    output_type: type[BaseModel] | None = None,
) -> dict[str, Any] | Command[Any]:
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

    if isinstance(tool.args_schema, type) and issubclass(tool.args_schema, BaseModel):
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
