"""Tool node factory wiring directly to LangGraph's ToolNode."""

from collections.abc import Sequence
from inspect import signature
from typing import Any, Awaitable, Callable, Literal

from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.types import Command
from pydantic import BaseModel
from uipath.platform.resume_triggers import is_no_content_marker
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)
from uipath_langchain.chat.hitl import check_tool_confirmation

# the type safety can be improved with generics
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

OutputType = dict[Literal["messages"], list[ToolMessage]] | Command[Any] | None


def _wrapper_needs_state(wrapper: ToolWrapperType | AsyncToolWrapperType) -> bool:
    """Check if wrapper function expects a state parameter."""
    params = list(signature(wrapper).parameters.values())
    return len(params) >= 3


class UiPathToolNode(RunnableCallable):
    """
    A ToolNode that can be used in a React agent graph.
    It extracts the tool call from the state messages and invokes the tool.
    It supports optional synchronous and asynchronous wrappers for custom processing.
    Generic over the state model.
    Args:
        tool: The tool to invoke.
        wrapper: An optional synchronous wrapper for custom processing.
        awrapper: An optional asynchronous wrapper for custom processing.

    Returns:
        A dict with ToolMessage or a Command.
    """

    def __init__(
        self,
        tool: BaseTool,
        wrapper: ToolWrapperType | None = None,
        awrapper: AsyncToolWrapperType | None = None,
        handle_tool_errors: bool = False,
    ):
        super().__init__(func=self._func, afunc=self._afunc, name=tool.name)
        self.tool = tool
        self.wrapper = wrapper
        self.awrapper = awrapper
        self.handle_tool_errors = handle_tool_errors

    def _func(self, state: AgentGraphState) -> OutputType:
        call = self._extract_tool_call(state)
        if call is None:
            return None

        confirmation = check_tool_confirmation(call, self.tool)
        if confirmation is not None and confirmation.cancelled:
            return self._process_result(call, confirmation.cancelled)

        try:
            if self.wrapper:
                inputs = self._prepare_wrapper_inputs(
                    self.wrapper, self.tool, call, state
                )
                result = self.wrapper(*inputs)
            else:
                result = self.tool.invoke(call)
            output = self._process_result(call, result)
            if confirmation is not None:
                confirmation.annotate_result(output)
            return output
        except Exception as e:
            if self.handle_tool_errors:
                return self._process_error_result(call, e)
            raise

    async def _afunc(self, state: AgentGraphState) -> OutputType:
        call = self._extract_tool_call(state)
        if call is None:
            return None

        confirmation = check_tool_confirmation(call, self.tool)
        if confirmation is not None and confirmation.cancelled:
            return self._process_result(call, confirmation.cancelled)

        try:
            if self.awrapper:
                inputs = self._prepare_wrapper_inputs(
                    self.awrapper, self.tool, call, state
                )
                result = await self.awrapper(*inputs)
            else:
                result = await self.tool.ainvoke(call)
            output = self._process_result(call, result)
            if confirmation is not None:
                confirmation.annotate_result(output)
            return output
        except Exception as e:
            if self.handle_tool_errors:
                return self._process_error_result(call, e)
            raise

    def _extract_tool_call(self, state: AgentGraphState) -> ToolCall | None:
        """Extract the tool call from the state messages."""

        latest_ai_message = find_latest_ai_message(state.messages)
        if latest_ai_message is None:
            return None

        try:
            current_tool_call_index = extract_current_tool_call_index(
                state.messages, self.tool.name
            )
        except AgentRuntimeError:
            # Handle cases where AIMessage has no tool calls or other invalid states
            return None

        if current_tool_call_index is None:
            return None

        return latest_ai_message.tool_calls[current_tool_call_index]

    def _process_error_result(self, call: ToolCall, error: Exception) -> OutputType:
        """Handle tool execution errors by creating an error ToolMessage."""
        error_message = ToolMessage(
            content=str(error),
            name=call["name"],
            tool_call_id=call["id"],
            status="error",
        )
        return {"messages": [error_message]}

    def _process_result(
        self, call: ToolCall, result: dict[str, Any] | Command[Any] | ToolMessage | None
    ) -> OutputType:
        """Process the tool result into a message format or return a Command.
        Strip NO_CONTENT markers into ToolMessages embedded in a Command.
        """
        if isinstance(result, Command):
            self._filter_result(result)
            return result
        elif isinstance(result, ToolMessage):
            if is_no_content_marker(result.content):
                result.content = ""
            return {"messages": [result]}
        else:
            content = "" if is_no_content_marker(result) else str(result)
            message = ToolMessage(
                content=content, name=call["name"], tool_call_id=call["id"]
            )
            return {"messages": [message]}

    @staticmethod
    def _filter_result(command: Command[Any]) -> None:
        """Strip NO_CONTENT markers from ToolMessages embedded in a Command."""
        update = getattr(command, "update", None)
        if not isinstance(update, dict):
            return
        messages = update.get("messages")
        if not messages:
            return
        for msg in messages:
            if isinstance(msg, ToolMessage) and is_no_content_marker(msg.content):
                msg.content = ""

    def _prepare_wrapper_inputs(
        self,
        wrapper: ToolWrapperType | AsyncToolWrapperType,
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> Sequence[Any]:
        """Prepare inputs for wrapper invocation based on its signature."""
        if _wrapper_needs_state(wrapper):
            filtered_state = self._filter_state(state, wrapper)
            return tool, call, filtered_state
        return tool, call

    def _filter_state(
        self, state: Any, wrapper: ToolWrapperType | AsyncToolWrapperType
    ) -> BaseModel:
        """Filter the state to the expected model type."""
        model_type = list(signature(wrapper).parameters.values())[2].annotation
        if not issubclass(model_type, BaseModel):
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.TOOL_INVALID_WRAPPER_STATE,
                title="Wrapper state parameter must be a pydantic BaseModel subclass.",
                detail=f"Got {model_type.__name__} instead of BaseModel for wrapper state parameter.",
                category=UiPathErrorCategory.SYSTEM,
            )
        return model_type.model_validate(state, from_attributes=True)


class ToolWrapperMixin:
    wrapper: ToolWrapperType | None = None
    awrapper: AsyncToolWrapperType | None = None

    def set_tool_wrappers(
        self,
        wrapper: ToolWrapperType | None = None,
        awrapper: AsyncToolWrapperType | None = None,
    ) -> None:
        """Define wrappers for the tool execution."""
        self.wrapper = wrapper
        self.awrapper = awrapper


def create_tool_node(
    tools: Sequence[BaseTool], handle_tool_errors: bool = False
) -> dict[str, UiPathToolNode]:
    """Create individual ToolNode for each tool.

    Args:
        tools: Sequence of tools to create nodes for.
        handle_tool_errors: If True, catch tool execution errors and return them as error ToolMessages
            instead of letting exceptions propagate.

    Returns:
        Dict mapping tool.name -> ReactToolNode([tool]).
        Each tool gets its own dedicated node for middleware composition.

    Note:
        handle_tool_errors=False delegates error handling to LangGraph's error boundary.
        handle_tool_errors=True will cause errors to be caught and converted to ToolMessages with status="error".
    """
    dict_mapping: dict[str, UiPathToolNode] = {}
    for tool in tools:
        if isinstance(tool, ToolWrapperMixin):
            dict_mapping[tool.name] = UiPathToolNode(
                tool,
                wrapper=tool.wrapper,
                awrapper=tool.awrapper,
                handle_tool_errors=handle_tool_errors,
            )
        else:
            dict_mapping[tool.name] = UiPathToolNode(
                tool, wrapper=None, awrapper=None, handle_tool_errors=handle_tool_errors
            )
    return dict_mapping
