"""For models that don't support forced tool choice (Opus 5.5).

Such a model can answer in text instead of calling end_execution or raise_error.
Then one more call binds only those two tools and asks the model to call one.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolCall
from langchain_core.tools import BaseTool
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL

from uipath_langchain.chat.thinking import is_reasoning_block

FINISH_REQUEST = (
    "Return the final output of the task by calling "
    + END_EXECUTION_TOOL.name
    + ", or call "
    + RAISE_ERROR_TOOL.name
    + " if the task failed."
)


def output_fields(tools: Sequence[BaseTool]) -> list[str] | None:
    """The agent's output fields (end_execution's arguments), or None."""
    for tool in tools:
        if tool.name == END_EXECUTION_TOOL.name:
            return list(tool.args)
    return None


def finish_without_forcing(
    tools: Sequence[BaseTool],
    messages: Sequence[AnyMessage],
    answer: AIMessage,
) -> tuple[list[BaseTool], list[AnyMessage]]:
    """The finish call's tools and messages: end_execution and raise_error only."""
    finishing_tools = [
        tool
        for tool in tools
        if tool.name in (END_EXECUTION_TOOL.name, RAISE_ERROR_TOOL.name)
    ]
    return finishing_tools, [*messages, answer, HumanMessage(content=FINISH_REQUEST)]


def finishing_tool_call(reply: AIMessage, output_fields: list[str]) -> AIMessage | None:
    """The reply's raise_error call, or its end_execution call if it has any output.

    The reply keeps its reasoning blocks: the message can be sent back to the model
    (an output file retry), and thinking models need them on a tool-use turn.
    """
    for tool_call in reply.tool_calls:
        if tool_call["name"] == RAISE_ERROR_TOOL.name:
            return _only(reply, tool_call)
    for tool_call in reply.tool_calls:
        if tool_call["name"] == END_EXECUTION_TOOL.name and _has_output(
            tool_call["args"], output_fields
        ):
            return _only(reply, tool_call)
    return None


def _has_output(arguments: dict[str, Any], fields: list[str]) -> bool:
    return not fields or any(
        arguments.get(field) not in (None, "", [], {}) for field in fields
    )


def _only(reply: AIMessage, tool_call: ToolCall) -> AIMessage:
    content = reply.content if isinstance(reply.content, list) else []
    reasoning = [block for block in content if is_reasoning_block(block)]
    return reply.model_copy(update={"content": reasoning, "tool_calls": [tool_call]})
