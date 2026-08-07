"""Shared tool wrapper types."""

from typing import Any, Awaitable, Callable

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from langgraph.types import Command

ToolWrapperReturnType = dict[str, Any] | Command[Any] | None

ToolWrapperWithoutState = Callable[[BaseTool, ToolCall], ToolWrapperReturnType]
ToolWrapperWithState = Callable[[BaseTool, ToolCall, Any], ToolWrapperReturnType]
ToolWrapperType = ToolWrapperWithoutState | ToolWrapperWithState

AsyncToolWrapperWithoutState = Callable[
    [BaseTool, ToolCall], Awaitable[ToolWrapperReturnType]
]
AsyncToolWrapperWithState = Callable[
    [BaseTool, ToolCall, Any], Awaitable[ToolWrapperReturnType]
]
AsyncToolWrapperType = AsyncToolWrapperWithoutState | AsyncToolWrapperWithState
