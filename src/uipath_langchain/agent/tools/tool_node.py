"""Tool node factory wiring directly to LangGraph's ToolNode."""

from collections.abc import Sequence
from inspect import signature
from typing import Any, Awaitable, Callable, Literal

from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.types import Command
from pydantic import BaseModel

from uipath_langchain.agent.exceptions import AgentStateException
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)

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
    ):
        super().__init__(func=self._func, afunc=self._afunc, name=tool.name)
        self.tool = tool
        self.wrapper = wrapper
        self.awrapper = awrapper

    def _func(self, state: AgentGraphState, config: RunnableConfig) -> OutputType:
        call = self._extract_tool_call(state)
        if call is None:
            return None
        if self.wrapper:
            inputs = self._prepare_wrapper_inputs(self.wrapper, self.tool, call, state)
            token = var_child_runnable_config.set(config)
            try:
                result = self.wrapper(*inputs)
            finally:
                var_child_runnable_config.reset(token)
        else:
            result = self.tool.invoke(call, config=config)
        return self._process_result(call, result)

    async def _afunc(
        self, state: AgentGraphState, config: RunnableConfig
    ) -> OutputType:
        call = self._extract_tool_call(state)
        if call is None:
            return None
        if self.awrapper:
            inputs = self._prepare_wrapper_inputs(self.awrapper, self.tool, call, state)
            token = var_child_runnable_config.set(config)
            try:
                result = await self.awrapper(*inputs)
            finally:
                var_child_runnable_config.reset(token)
        else:
            result = await self.tool.ainvoke(call, config=config)
        return self._process_result(call, result)

    def _extract_tool_call(self, state: AgentGraphState) -> ToolCall | None:
        """Extract the tool call from the state messages."""

        latest_ai_message = find_latest_ai_message(state.messages)
        if latest_ai_message is None:
            return None

        try:
            current_tool_call_index = extract_current_tool_call_index(
                state.messages, self.tool.name
            )
        except AgentStateException:
            # Handle cases where AIMessage has no tool calls or other invalid states
            return None

        if current_tool_call_index is None:
            return None

        return latest_ai_message.tool_calls[current_tool_call_index]

    def _process_result(
        self, call: ToolCall, result: dict[str, Any] | Command[Any] | ToolMessage | None
    ) -> OutputType:
        """Process the tool result into a message format or return a Command."""
        if isinstance(result, Command):
            return result
        elif isinstance(result, ToolMessage):
            return {"messages": [result]}
        else:
            message = ToolMessage(
                content=str(result), name=call["name"], tool_call_id=call["id"]
            )
            return {"messages": [message]}

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
            raise ValueError(
                "Wrapper state parameter must be a pydantic BaseModel subclass."
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


def create_tool_node(tools: Sequence[BaseTool]) -> dict[str, UiPathToolNode]:
    """Create individual ToolNode for each tool.

    Args:
        tools: Sequence of tools to create nodes for.

    Returns:
        Dict mapping tool.name -> ReactToolNode([tool]).
        Each tool gets its own dedicated node for middleware composition.

    Note:
        handle_tool_errors=False delegates error handling to LangGraph's error boundary.
    """
    dict_mapping: dict[str, UiPathToolNode] = {}
    for tool in tools:
        if isinstance(tool, ToolWrapperMixin):
            dict_mapping[tool.name] = UiPathToolNode(
                tool, wrapper=tool.wrapper, awrapper=tool.awrapper
            )
        else:
            dict_mapping[tool.name] = UiPathToolNode(tool, wrapper=None, awrapper=None)
    return dict_mapping
