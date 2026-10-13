"""The advanced agent loop honors the configured iteration budget."""

import asyncio
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.advanced.agent import (
    _SUBAGENT_ITERATIONS_KEY,
    _IterationBudget,
    _MaxIterationsMiddleware,
    _SubagentIterationsMiddleware,
    _with_subagent_spend,
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


def _ping_forever(calls: list[int]) -> Iterator[AIMessage]:
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


def _never_stops(calls: list[int]) -> _ToolCallingFakeModel:
    """A model that always asks for another tool call, so only the budget stops it."""
    return _ToolCallingFakeModel(messages=_ping_forever(calls))


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


def _delegates_then_never_stops(calls: list[int]) -> _ToolCallingFakeModel:
    """A model whose first call delegates to a subagent that then loops forever."""

    def messages() -> Iterator[AIMessage]:
        calls.append(1)
        yield AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {
                        "description": "loop",
                        "subagent_type": "general-purpose",
                    },
                    "id": "delegate",
                }
            ],
        )
        yield from _ping_forever(calls)

    return _ToolCallingFakeModel(messages=messages())


@pytest.mark.asyncio
async def test_subagent_calls_spend_the_turn_budget() -> None:
    calls: list[int] = []
    graph = _autonomous_graph(_delegates_then_never_stops(calls), max_iterations=3)

    with pytest.raises(AgentRuntimeError) as error:
        await graph.ainvoke({"task": "delegate"}, {"recursion_limit": 100})

    assert error.value.error_info.code == "AGENT_RUNTIME.TERMINATION_MAX_ITERATIONS"
    assert len(calls) == 3


class _DelegatingModel(_ToolCallingFakeModel):
    """Delegates on the user's request, then asks for another tool call forever.

    ``subagent_answers`` makes the subagent reply on its first call, and
    ``main_answers`` makes the main agent reply once the subagent reported back.
    """

    calls: list[int] = []
    subagent_answers: bool = False
    main_answers: bool = False

    def _generate(
        self, messages: list[BaseMessage], *args: Any, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(1)
        last = messages[-1]
        if isinstance(last, HumanMessage) and last.content == "delegate":
            call: dict[str, Any] = {
                "name": "task",
                "args": {"description": "loop", "subagent_type": "general-purpose"},
            }
        elif (isinstance(last, HumanMessage) and self.subagent_answers) or (
            isinstance(last, ToolMessage) and last.name == "task" and self.main_answers
        ):
            return ChatResult(generations=[ChatGeneration(message=AIMessage("done"))])
        else:
            call = {"name": "ping", "args": {"value": "x"}}
        message = AIMessage(
            content="", tool_calls=[{**call, "id": f"call-{len(self.calls)}"}]
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.mark.asyncio
async def test_budget_is_kept_per_thread() -> None:
    """Concurrent runs sharing a compiled graph each get the whole budget."""
    model = _DelegatingModel(messages=iter([]), calls=[])
    graph = _autonomous_graph(model, max_iterations=3)

    async def run(thread_id: str) -> None:
        with pytest.raises(AgentRuntimeError):
            await graph.ainvoke(
                {"task": "delegate"},
                {"recursion_limit": 100, "configurable": {"thread_id": thread_id}},
            )

    await asyncio.gather(run("a"), run("b"))

    assert len(model.calls) == 6


@pytest.mark.asyncio
async def test_finished_subagent_calls_still_count() -> None:
    """A subagent's calls keep counting after its task call returns."""
    model = _DelegatingModel(messages=iter([]), calls=[], subagent_answers=True)
    graph = _autonomous_graph(model, max_iterations=4)

    with pytest.raises(AgentRuntimeError):
        await graph.ainvoke({"task": "delegate"}, {"recursion_limit": 100})

    assert len(model.calls) == 4


@pytest.mark.asyncio
async def test_each_exchange_gets_a_fresh_subagent_budget() -> None:
    model = _DelegatingModel(
        messages=iter([]), calls=[], subagent_answers=True, main_answers=True
    )
    graph = create_conversational_advanced_agent_graph(
        model=model,
        tools=[ping],
        system_prompt="sys",
        backend=None,
        max_iterations=3,
    ).compile(checkpointer=InMemorySaver())
    config: Any = {"recursion_limit": 100, "configurable": {"thread_id": "chat"}}

    await graph.ainvoke({"messages": [HumanMessage("delegate")]}, config)
    state = await graph.aget_state(config)
    history = [*state.values["messages"], HumanMessage("delegate")]
    await graph.ainvoke({"messages": history}, config)

    assert len(model.calls) == 6


def test_running_subagents_are_forgotten_once_their_task_ends() -> None:
    budget = _IterationBudget(max_iterations=5)
    with patch(
        "uipath_langchain.agent.advanced.agent.get_config",
        return_value={"configurable": {"thread_id": "t"}},
    ):
        budget.start_task(spent=1)
        budget.start_task(spent=1)
        budget.spend_subagent()
        assert budget.finish_task() == 1
        budget.spend_subagent()
        assert budget.finish_task() == 1

    assert budget._running == {}


def test_sync_hooks_share_the_budget() -> None:
    """The sync hooks charge the same budget as the async ones."""
    budget = _IterationBudget(max_iterations=3)
    main = _MaxIterationsMiddleware(budget)
    subagent = _SubagentIterationsMiddleware(budget)
    state = {"messages": [HumanMessage("delegate"), AIMessage("", id="a1")]}
    answer = ToolMessage(content="done", tool_call_id="t1")

    def run_subagent(request: Any) -> ToolMessage:
        subagent.wrap_model_call(MagicMock(), lambda _: MagicMock())
        subagent.wrap_model_call(MagicMock(), lambda _: MagicMock())
        with pytest.raises(AgentRuntimeError):
            subagent.wrap_model_call(MagicMock(), lambda _: MagicMock())
        return answer

    with patch(
        "uipath_langchain.agent.advanced.agent.get_config",
        return_value={"configurable": {"thread_id": "t"}},
    ):
        response = MagicMock()
        assert (
            main.wrap_model_call(MagicMock(state=state), lambda _: response) is response
        )
        ping_request = MagicMock(tool_call={"name": "ping"}, state=state)
        assert main.wrap_tool_call(ping_request, lambda _: answer) is answer
        task_request = MagicMock(tool_call={"name": "task"}, state=state)
        result = main.wrap_tool_call(task_request, run_subagent)

    assert result == Command(update={"messages": [answer], _SUBAGENT_ITERATIONS_KEY: 2})
    assert budget._running == {}


def test_a_subagent_outside_a_task_call_is_not_charged() -> None:
    budget = _IterationBudget(max_iterations=1)
    with patch(
        "uipath_langchain.agent.advanced.agent.get_config",
        return_value={"configurable": {"thread_id": "t"}},
    ):
        budget.spend_subagent()

    assert budget._running == {}


def test_subagent_spend_rides_on_any_task_result() -> None:
    message = ToolMessage(content="no such subagent", tool_call_id="t1")

    assert _with_subagent_spend(message, 0) is message
    assert _with_subagent_spend(message, 2) == Command(
        update={"messages": [message], _SUBAGENT_ITERATIONS_KEY: 2}
    )
    with_list_update: Command[Any] = Command(update=[("messages", [message])])
    assert _with_subagent_spend(with_list_update, 2) is with_list_update
