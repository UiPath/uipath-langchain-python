"""The advanced agent loop honors the configured iteration budget."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.advanced.agent import (
    _MaxIterationsMiddleware,
    create_advanced_agent_graph,
    create_conversational_advanced_agent_graph,
)
from uipath_langchain.agent.exceptions import AgentRuntimeError


class _Input(BaseModel):
    task: str = ""


class _Output(BaseModel):
    result: str = ""


@tool
def ping(value: str) -> str:
    """Echo the value back."""
    return value


class _ToolCallingFakeModel(GenericFakeChatModel):
    """A fake model that accepts tool bindings, so the real loop can run."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self


def _never_stops(calls: list[int]) -> _ToolCallingFakeModel:
    """A model that always asks for another tool call, so only the budget stops it."""

    def messages() -> Iterator[AIMessage]:
        while True:
            calls.append(len(calls) + 1)
            yield AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ping",
                        "args": {"value": str(len(calls))},
                        "id": f"call-{len(calls)}",
                    }
                ],
            )

    return _ToolCallingFakeModel(messages=messages())


def _autonomous_graph(model: BaseChatModel, max_iterations: int | None) -> Any:
    return create_advanced_agent_graph(
        model=model,
        tools=[ping],
        system_prompt="sys",
        backend=None,
        response_format=None,
        input_schema=_Input,
        output_schema=_Output,
        build_user_message=lambda args: args.get("task", ""),
        max_iterations=max_iterations,
    ).compile()


@pytest.mark.asyncio
async def test_autonomous_loop_stops_at_max_iterations() -> None:
    calls: list[int] = []
    graph = _autonomous_graph(_never_stops(calls), max_iterations=3)

    with pytest.raises(AgentRuntimeError) as error:
        await graph.ainvoke({"task": "loop"}, {"recursion_limit": 100})

    assert error.value.error_info.code == "AGENT_RUNTIME.TERMINATION_MAX_ITERATIONS"
    assert error.value.error_info.title == "Maximum iterations of '3' reached."
    assert error.value.error_info.category == UiPathErrorCategory.USER
    assert len(calls) == 3


def test_no_middleware_without_a_limit() -> None:
    with patch(
        "uipath_langchain.agent.advanced.agent._create_deep_agent",
        return_value=MagicMock(),
    ) as mock_create:
        _autonomous_graph(MagicMock(spec=BaseChatModel), max_iterations=None)

    middleware = mock_create.call_args.kwargs["middleware"]
    assert not any(isinstance(m, _MaxIterationsMiddleware) for m in middleware)


@pytest.mark.asyncio
async def test_conversational_budget_is_per_exchange() -> None:
    """Messages carried in from earlier exchanges do not spend this exchange's budget."""
    calls: list[int] = []
    graph = create_conversational_advanced_agent_graph(
        model=_never_stops(calls),
        tools=[ping],
        system_prompt="sys",
        backend=None,
        max_iterations=2,
    ).compile()

    history: list[Any] = [
        HumanMessage(content="hi", id="u1"),
        AIMessage(content="hello", id="a1"),
        AIMessage(content="still here", id="a2"),
        HumanMessage(content="keep going", id="u2"),
    ]

    with pytest.raises(AgentRuntimeError):
        await graph.ainvoke({"messages": history}, {"recursion_limit": 100})

    assert len(calls) == 2
