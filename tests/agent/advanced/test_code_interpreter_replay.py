"""A tool reached from inside ``eval`` must not run twice across a suspend.

An interrupt raised while an ``eval`` is still executing does not resume the
``eval`` where it stopped. LangGraph replays the tool node from its checkpoint, so
the ``eval`` re-runs from the top and every bridged call it already made fires a
second time. That is why ``ptc`` withholds suspending tools, and why ``task()`` is
withheld whenever a subagent can suspend: ``task()`` reaches a subagent's tools
through a path the ``ptc`` allowlist does not cover.

These tests drive a real WASM guest through a real interrupt and resume. The
forced case is what the derived one has to prevent, so it is asserted rather than
described: without it, a change that re-exposes ``task()`` would look harmless.
"""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from deepagents import SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from uipath_langchain._utils.durable_interrupt import SUSPENDS_RUN
from uipath_langchain.agent.advanced import build_code_interpreter_middleware

pytest.importorskip("langchain_quickjs", reason="needs the code-interpreter extra")

_CODE = """
await tools.audit({ note: "before-task" });
const r = await task({ description: "ask", subagentType: "worker" });
"done: " + JSON.stringify(r)
"""


class _ScriptedModel(GenericFakeChatModel):
    """Replays scripted messages and ignores the bound tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


def _audit_tool(sink: list[str]) -> BaseTool:
    def audit(note: str = "") -> str:
        """Record that this ran."""
        sink.append(note)
        return f"recorded {note}"

    return StructuredTool.from_function(func=audit, name="audit", description="record")


def _escalation_tool() -> BaseTool:
    def escalate(question: str = "") -> str:
        """Ask a human."""
        return f"human said: {interrupt({'question': question})}"

    return StructuredTool.from_function(
        func=escalate,
        name="escalate",
        description="ask a human",
        metadata={SUSPENDS_RUN: True},
    )


def _run_until_suspend_then_resume(
    workspace: Path, middleware: list[Any], sink: list[str]
) -> list[str]:
    """Run an agent whose subagent escalates mid-``eval``, then resume it."""
    main = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "eval", "args": {"code": _CODE}, "id": "c1"}],
                ),
                AIMessage(content="finished"),
            ]
        )
    )
    sub = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "escalate", "args": {"question": "ok?"}, "id": "s1"}
                    ],
                ),
                AIMessage(content="sub done"),
            ]
            * 2
        )
    )
    graph = create_deep_agent(
        model=main,
        tools=[_audit_tool(sink)],
        subagents=[
            cast(
                "SubAgent",
                {
                    "name": "worker",
                    "description": "escalates",
                    "system_prompt": "escalate",
                    "tools": [_escalation_tool()],
                    "model": sub,
                },
            )
        ],
        backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "replay-test"}}
    result = asyncio.run(
        graph.ainvoke({"messages": [{"role": "user", "content": "go"}]}, config)
    )
    assert result.get("__interrupt__"), "the subagent did not suspend the run"
    asyncio.run(graph.ainvoke(Command(resume="yes"), config))
    return sink


def test_forcing_task_into_the_repl_duplicates_a_bridged_call(tmp_path: Path) -> None:
    """The failure the derivation exists to prevent, asserted rather than assumed."""
    from langchain_quickjs import CodeInterpreterMiddleware

    sink: list[str] = []
    middleware = [CodeInterpreterMiddleware(ptc=["audit"], subagents=True)]

    assert _run_until_suspend_then_resume(tmp_path, middleware, sink) == [
        "before-task",
        "before-task",
    ]


def test_a_suspending_subagent_leaves_task_out_of_the_repl(tmp_path: Path) -> None:
    """With ``task()`` withheld the ``eval`` cannot suspend, so nothing replays."""
    sink: list[str] = []
    # The subagent declares no tools, so it inherits this list, escalation included.
    tools = [_audit_tool(sink), _escalation_tool()]
    middleware = build_code_interpreter_middleware(
        tools,
        subagents=[
            cast(
                "SubAgent",
                {"name": "worker", "description": "escalates", "system_prompt": "esc"},
            )
        ],
    )
    main = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "eval", "args": {"code": _CODE}, "id": "c1"}],
                ),
                AIMessage(content="finished"),
            ]
        )
    )
    graph = create_deep_agent(
        model=main,
        tools=tools,
        subagents=[
            cast(
                "SubAgent",
                {
                    "name": "worker",
                    "description": "escalates",
                    "system_prompt": "esc",
                    "model": main,
                },
            )
        ],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )
    result = asyncio.run(
        graph.ainvoke(
            {"messages": [{"role": "user", "content": "go"}]},
            cast("RunnableConfig", {"configurable": {"thread_id": "no-task"}}),
        )
    )
    assert not result.get("__interrupt__"), "the eval suspended despite no task()"
    assert sink == ["before-task"]
    output = next(m.content for m in result["messages"] if m.type == "tool")
    assert "task is not defined" in str(output)
