"""Tests for voice stub graph builder."""

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START
from langgraph.types import Command, interrupt

from uipath_agents.voice.graph import build_voice_tool_graph


@tool
def add_numbers(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def failing_tool(query: str) -> str:
    """A tool that always fails."""
    raise ValueError("Something went wrong")


@tool
def interrupting_tool(process_name: str) -> str:
    """A tool that calls interrupt()."""
    result = interrupt({"process_name": process_name})
    return f"Completed: {result}"


class TestBuildVoiceToolGraph:
    """Test graph structure and compilation."""

    def test_graph_has_single_node(self) -> None:
        graph = build_voice_tool_graph(add_numbers)
        assert list(graph.nodes.keys()) == ["add_numbers"]

    def test_graph_has_correct_edges(self) -> None:
        graph = build_voice_tool_graph(add_numbers)
        edges = graph._all_edges
        assert (START, "add_numbers") in edges
        assert ("add_numbers", END) in edges

    def test_graph_compiles_with_checkpointer(self) -> None:
        graph = build_voice_tool_graph(add_numbers)
        compiled = graph.compile(checkpointer=MemorySaver())
        assert compiled is not None


@pytest.mark.asyncio
class TestGraphExecution:
    """Test graph execution for all tool outcome types."""

    @staticmethod
    def _make_input(
        tool_name: str, args: dict[str, Any], call_id: str = "test-call"
    ) -> dict[str, Any]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": call_id, "name": tool_name, "args": args}],
                )
            ]
        }

    async def test_successful_tool_completes_graph(self) -> None:
        graph = build_voice_tool_graph(add_numbers)
        compiled = graph.compile(checkpointer=MemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

        result = await compiled.ainvoke(
            self._make_input("add_numbers", {"a": 3, "b": 7}),  # type: ignore[call-overload]
            config,
        )

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].type == "ai"
        assert messages[1].type == "tool"
        assert messages[1].content == "10"

    async def test_tool_exception_propagates_with_handle_errors_false(self) -> None:
        graph = build_voice_tool_graph(failing_tool)
        compiled = graph.compile(checkpointer=MemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": "t2"}}

        with pytest.raises(Exception, match="Something went wrong"):
            await compiled.ainvoke(
                self._make_input("failing_tool", {"query": "test"}),  # type: ignore[call-overload]
                config,
            )

    async def test_interrupt_suspends_graph(self) -> None:
        graph = build_voice_tool_graph(interrupting_tool)
        compiled = graph.compile(checkpointer=MemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": "t3"}}

        result = await compiled.ainvoke(
            self._make_input("interrupting_tool", {"process_name": "Invoice"}),  # type: ignore[call-overload]
            config,
        )

        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0].type == "ai"

        state = await compiled.aget_state(config)
        assert state.next, "Graph should have pending next nodes (suspended)"
        assert len(state.interrupts) == 1
        assert state.interrupts[0].value == {"process_name": "Invoice"}

    async def test_interrupt_then_resume_completes(self) -> None:
        graph = build_voice_tool_graph(interrupting_tool)
        compiled = graph.compile(checkpointer=MemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": "t4"}}

        await compiled.ainvoke(
            self._make_input("interrupting_tool", {"process_name": "Invoice"}),  # type: ignore[call-overload]
            config,
        )

        state = await compiled.aget_state(config)
        assert state.next, "Graph should be suspended"

        result = await compiled.ainvoke(Command(resume="process_output_data"), config)

        messages = result["messages"]
        tool_messages = [m for m in messages if m.type == "tool"]
        assert len(tool_messages) == 1
        assert "Completed: process_output_data" in tool_messages[0].content

        state = await compiled.aget_state(config)
        assert not state.next, "Graph should be completed after resume"
