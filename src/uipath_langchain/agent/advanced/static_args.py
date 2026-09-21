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

from deepagents.middleware.async_subagents import AsyncSubAgentState
from deepagents.middleware.filesystem import FilesystemState
from deepagents.middleware.memory import MemoryState
from deepagents.middleware.rubric import RubricState
from deepagents.middleware.skills import SkillsState
from deepagents.middleware.summarization import SummarizationState
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
    agent_input_from_state,
    has_argument_bindings,
)

logger = logging.getLogger(__name__)

# Channels declared by the deep agent's own middleware. Declaring an agent input
# under one of these names would replace the channel (and its reducer) rather
# than add a key.
_RESERVED_STATE_KEYS: frozenset[str] = frozenset(
    {
        *AgentState.__annotations__,
        *PlanningState.__annotations__,
        *FilesystemState.__annotations__,
        *SkillsState.__annotations__,
        *SummarizationState.__annotations__,
        *MemoryState.__annotations__,
        *AsyncSubAgentState.__annotations__,
        *RubricState.__annotations__,
    }
)


def _is_reserved(name: str) -> bool:
    # deepagents keeps its private channels underscored (summarization, rubric,
    # forked context), so the prefix is reserved wholesale.
    return name in _RESERVED_STATE_KEYS or name.startswith("_")


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
    subagent carrying this middleware resolves the same bindings. The input is
    read from the state on every model call and the bindings are re-resolved
    whenever it changes, so a compiled graph invoked again with other input, or
    resumed from a checkpoint, pins the values of that invocation.
    """

    def __init__(self, input_schema: type[BaseModel] | None) -> None:
        self._handler = StaticArgsHandler()

        declared = list((input_schema or BaseModel).model_fields)
        self._input_fields = [name for name in declared if not _is_reserved(name)]
        reserved = sorted(set(declared) - set(self._input_fields))
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

    def _prepare_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        bound_tools = [tool for tool in request.tools if isinstance(tool, BaseTool)]
        if not any(has_argument_bindings(tool) for tool in bound_tools):
            return request

        agent_input = agent_input_from_state(
            cast(Mapping[str, Any], request.state), self._input_fields
        )
        pinned_by_name = {
            original.name: pinned
            for original, pinned in zip(
                bound_tools,
                self._handler.resolve(bound_tools, agent_input),
                strict=True,
            )
            if pinned is not original
        }
        if not pinned_by_name:
            return request
        return request.override(
            tools=[
                pinned_by_name.get(tool.name, tool)
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
