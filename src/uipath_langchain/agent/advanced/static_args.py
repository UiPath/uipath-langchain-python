"""Configured tool argument bindings for advanced agents.

A low-code tool can bind an argument to a static value, an agent input, or a
text or array built from inputs (``argument_properties`` on the tool). The
standard ReAct llm node applies those bindings around every model call with
``StaticArgsHandler``: it binds the model to schemas that pin the bound fields
and writes the resolved values into the returned tool calls. ``create_deep_agent``
binds tools as given, so on the advanced path the model would be free to fill a
bound field with anything and nothing would overwrite it. This middleware runs
the same handler at the deep agent's model-call boundary, on the main agent and
on every subagent.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, NotRequired, cast

from deepagents.middleware.filesystem import FilesystemState
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.todo import PlanningState
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from uipath_langchain.agent.tools.static_args import (
    StaticArgsHandler,
    has_argument_bindings,
)

logger = logging.getLogger(__name__)

# Channels the deep agent already owns. Declaring an agent input under one of
# these names would replace the channel (and its reducer) rather than add a key.
_RESERVED_STATE_KEYS: frozenset[str] = frozenset(
    {
        *AgentState.__annotations__,
        *FilesystemState.__annotations__,
        *PlanningState.__annotations__,
    }
)


def build_static_args_middleware(
    tools: Sequence[BaseTool],
    input_schema: type[BaseModel] | None,
) -> list[AgentMiddleware[Any, Any]]:
    """The static-args middleware for ``tools``, ready to splice into ``middleware``.

    Empty when no tool carries bindings, so an agent that has none keeps the
    deep agent's state untouched.
    """
    if not any(has_argument_bindings(tool) for tool in tools):
        return []
    return [StaticArgsMiddleware(input_schema)]


class StaticArgsMiddleware(AgentMiddleware[AgentState[Any], Any]):
    """Apply configured tool argument bindings around every deep-agent model call.

    Bindings to agent inputs need the invocation's input, which lives on the
    wrapper graph's state. Declaring the input fields on ``state_schema`` is what
    carries them into the deep agent's state, where ``request.state`` exposes
    them; deepagents copies that state into each subagent it dispatches, so a
    subagent carrying this middleware resolves the same bindings. Bindings are
    resolved once, on the first model call, the way the standard llm node does;
    a resumed run resolves them again from the checkpointed state.
    """

    def __init__(self, input_schema: type[BaseModel] | None) -> None:
        self._input_schema: type[BaseModel] = input_schema or BaseModel
        self._handler = StaticArgsHandler()
        self._schema_tools_by_name: dict[str, BaseTool] | None = None

        self._input_fields = [
            name
            for name in self._input_schema.model_fields
            if name not in _RESERVED_STATE_KEYS
        ]
        reserved = sorted(
            set(self._input_schema.model_fields) - set(self._input_fields)
        )
        if reserved:
            logger.warning(
                "Agent inputs %s share a name with deep-agent state and cannot be "
                "bound to tool arguments in Advanced Mode.",
                reserved,
            )
        self.state_schema = cast(
            type[AgentState[Any]],
            type(
                "StaticArgsState",
                (AgentState,),
                {
                    "__annotations__": {
                        name: NotRequired[Any] for name in self._input_fields
                    }
                },
            ),
        )

    def _agent_input(self, state: Mapping[str, Any]) -> BaseModel:
        values = {name: state[name] for name in self._input_fields if name in state}
        return self._input_schema.model_validate(values, from_attributes=True)

    def _schema_tools(self, request: ModelRequest[Any]) -> dict[str, BaseTool]:
        """Tools whose model-facing schema pins a bound field, by name."""
        if self._schema_tools_by_name is None:
            bound_tools = [tool for tool in request.tools if isinstance(tool, BaseTool)]
            processed = self._handler.initialize(
                bound_tools,
                self._agent_input(cast(Mapping[str, Any], request.state)),
                self._input_schema,
            )
            self._schema_tools_by_name = {
                original.name: modified
                for original, modified in zip(bound_tools, processed, strict=True)
                if modified is not original
            }
        return self._schema_tools_by_name

    def _prepare_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        schema_tools = self._schema_tools(request)
        if not schema_tools:
            return request
        return request.override(
            tools=[
                schema_tools.get(tool.name, tool)
                if isinstance(tool, BaseTool)
                else tool
                for tool in request.tools
            ]
        )

    def _apply_to_response(self, response: ModelResponse[Any]) -> None:
        for message in response.result:
            if isinstance(message, AIMessage) and message.tool_calls:
                self._handler.apply_to_response(message.tool_calls)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        response = handler(self._prepare_request(request))
        self._apply_to_response(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        response = await handler(self._prepare_request(request))
        self._apply_to_response(response)
        return response
