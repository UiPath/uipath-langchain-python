"""Tool execution errors reach the model instead of faulting the run.

Not mocked: the behaviour lives in how deepagents composes our middleware around
its tool node, so only a real graph exercises it.
"""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.advanced.agent import create_advanced_agent
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)


class _Model(GenericFakeChatModel):
    model_name: str = "test-model"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return {"ls_provider": "openai", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_Model":
        return self


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _failing_tool(error: Callable[[], Exception], calls: list[str]) -> BaseTool:
    def get_weather(city: str) -> str:
        calls.append(city)
        raise error()

    return StructuredTool.from_function(
        func=get_weather, name="get_weather", description="Get the weather."
    )


def _run(tool: BaseTool, script: list[AIMessage]) -> dict[str, Any]:
    model = _Model(messages=iter([*script, *[AIMessage(content="done")] * 10]))
    graph = create_advanced_agent(model=model, tools=[tool])
    return asyncio.run(
        graph.ainvoke({"messages": [{"role": "user", "content": "weather?"}]})
    )


def test_tool_exception_is_returned_to_the_model() -> None:
    calls: list[str] = []
    tool = _failing_tool(lambda: RuntimeError("upstream 503"), calls)

    result = _run(tool, [_call("get_weather", {"city": "Paris"}, "c1")])

    errors = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "c1"
    ]
    assert calls == ["Paris"]
    assert [(m.status, m.content) for m in errors] == [("error", "upstream 503")]
    assert result["messages"][-1].content == "done"


@pytest.mark.parametrize(
    "error",
    [
        lambda: AgentRuntimeError(
            code=AgentRuntimeErrorCode.TERMINATION_ESCALATION_REJECTED,
            title="rejected",
            detail="rejected",
            category=UiPathErrorCategory.USER,
        ),
        lambda: AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="bad config",
            detail="bad config",
            category=UiPathErrorCategory.USER,
        ),
    ],
    ids=["termination", "startup"],
)
def test_errors_that_end_the_run_still_fault_it(
    error: Callable[[], Exception],
) -> None:
    tool = _failing_tool(error, [])

    with pytest.raises(type(error())):
        _run(tool, [_call("get_weather", {"city": "Paris"}, "c1")])


def test_non_termination_agent_runtime_error_is_returned_to_the_model() -> None:
    tool = _failing_tool(
        lambda: AgentRuntimeError(
            code=AgentRuntimeErrorCode.HTTP_ERROR,
            title="timed out",
            detail="timed out",
            category=UiPathErrorCategory.SYSTEM,
            should_wrap=False,
        ),
        [],
    )

    result = _run(tool, [_call("get_weather", {"city": "Paris"}, "c1")])

    assert result["messages"][-1].content == "done"


def test_subagent_tool_exception_is_returned_to_the_subagent() -> None:
    calls: list[str] = []
    tool = _failing_tool(lambda: RuntimeError("upstream 503"), calls)

    result = _run(
        tool,
        [
            _call(
                "task",
                {"description": "go", "subagent_type": "general-purpose"},
                "t1",
            ),
            _call("get_weather", {"city": "Paris"}, "c1"),
        ],
    )

    task_results = [
        m
        for m in result["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "t1"
    ]
    assert calls == ["Paris"]
    assert [m.status for m in task_results] == ["success"]


def test_interrupt_still_suspends_and_resumes() -> None:
    def approve(city: str) -> str:
        return f"approved {interrupt({'city': city})}"

    tool = StructuredTool.from_function(
        func=approve, name="approve", description="Ask for approval."
    )
    model = _Model(
        messages=iter(
            [_call("approve", {"city": "Paris"}, "c1"), AIMessage(content="done")]
        )
    )
    graph = create_advanced_agent(model=model, tools=[tool])
    graph.checkpointer = InMemorySaver()
    config: Any = {"configurable": {"thread_id": "t"}}

    suspended = asyncio.run(
        graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, config)
    )
    resumed = asyncio.run(graph.ainvoke(Command(resume="yes"), config))

    results = [
        m
        for m in resumed["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "c1"
    ]
    assert suspended["__interrupt__"][0].value == {"city": "Paris"}
    assert [(m.status, m.content) for m in results] == [("success", "approved yes")]
