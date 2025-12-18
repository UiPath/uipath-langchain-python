"""Tool node factory wiring directly to LangGraph's ToolNode."""

from collections.abc import Sequence
from inspect import signature
from typing import Any, Awaitable, Callable, Literal

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.types import Command
from pydantic import BaseModel

# the type safety can be improved with generics
ToolWrapperType = Callable[
    [BaseTool, ToolCall, Any], dict[str, Any] | Command[Any] | None
]
AsyncToolWrapperType = Callable[
    [BaseTool, ToolCall, Any],
    Awaitable[dict[str, Any] | Command[Any] | None],
]
OutputType = dict[Literal["messages"], list[ToolMessage]] | Command[Any] | None


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

    def _func(self, state: Any) -> OutputType:
        call = self._extract_tool_call(state)
        if call is None:
            return None
        if self.wrapper:
            filtered_state = self._filter_state(state, self.wrapper)
            result = self.wrapper(self.tool, call, filtered_state)
        else:
            result = self.tool.invoke(call["args"])

        return self._process_result(call, result)

    async def _afunc(self, state: Any) -> OutputType:
        call = self._extract_tool_call(state)
        if call is None:
            return None
        if self.awrapper:
            filtered_state = self._filter_state(state, self.awrapper)
            result = await self.awrapper(self.tool, call, filtered_state)
        else:
            result = await self.tool.ainvoke(call["args"])

        return self._process_result(call, result)

    def _extract_tool_call(self, state: Any) -> ToolCall | None:
        """Extract the tool call from the state messages."""

        if not hasattr(state, "messages"):
            raise ValueError("State does not have messages key")

        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError("Last message in message stack is not an AIMessage.")

        for tool_call in last_message.tool_calls:
            if tool_call["name"] == self.tool.name:
                return tool_call
        return None

    def _process_result(
        self, call: ToolCall, result: dict[str, Any] | Command[Any] | None
    ) -> OutputType:
        """Process the tool result into a message format or return a Command."""
        if isinstance(result, Command):
            return result
        else:
            message = ToolMessage(
                content=str(result), name=call["name"], tool_call_id=call["id"]
            )
            return {"messages": [message]}

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
        agentState: The type of the agent state model.

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
