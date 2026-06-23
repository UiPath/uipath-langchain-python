"""Tests for client-side tool filtering logic."""

import asyncio
from collections.abc import Awaitable
from typing import Any

from uipath_langchain.agent.tools.client_side_tool import (
    ClientSideToolInfo,
    apply_tool_filter,
    available_client_side_tools,
)

AGENT_TOOLS: dict[str, ClientSideToolInfo] = {
    "get_weather": {
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
        "output_schema": {
            "type": "object",
            "properties": {"temp": {"type": "number"}},
        },
    },
    "show_map": {
        "input_schema": None,
        "output_schema": None,
    },
}


def _run_sync(awaitable: Awaitable[Any]) -> Any:
    """Run a coroutine on a fresh event loop.

    Avoids asyncio.get_event_loop() raising 'no current event loop' once an
    earlier pytest-asyncio test has closed the thread's loop.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


class TestApplyToolFilter:
    """Tests for apply_tool_filter."""

    def test_all_agent_tools_declared(self):
        apply_tool_filter(["get_weather", "show_map"], AGENT_TOOLS)
        assert available_client_side_tools.get() == {"get_weather", "show_map"}

    def test_subset_of_agent_tools(self):
        """Client passes [A] when agent has [A, B] — only A available."""
        apply_tool_filter(["get_weather"], AGENT_TOOLS)
        assert available_client_side_tools.get() == {"get_weather"}

    def test_empty_list_means_no_tools(self):
        """Client passes [] — no tools available."""
        apply_tool_filter([], AGENT_TOOLS)
        assert available_client_side_tools.get() == set()

    def test_unknown_names_ignored(self):
        """Client passes [A, C, B] when agent has [A, B] — only [A, B]."""
        apply_tool_filter(["get_weather", "unknown_tool", "show_map"], AGENT_TOOLS)
        assert available_client_side_tools.get() == {"get_weather", "show_map"}

    def test_only_unknown_names(self):
        """Client passes only unknown names — empty set."""
        apply_tool_filter(["foo", "bar"], AGENT_TOOLS)
        assert available_client_side_tools.get() == set()

    def test_dict_declarations_with_name(self):
        """Dicts with 'name' field are accepted."""
        apply_tool_filter([{"name": "get_weather"}, {"name": "show_map"}], AGENT_TOOLS)
        assert available_client_side_tools.get() == {"get_weather", "show_map"}

    def test_mixed_strings_and_dicts(self):
        apply_tool_filter(["get_weather", {"name": "show_map"}], AGENT_TOOLS)
        assert available_client_side_tools.get() == {"get_weather", "show_map"}

    def test_dicts_without_name_silently_skipped(self):
        """Dicts missing 'name' are skipped, not errored."""
        apply_tool_filter([{"inputSchema": {}}, "get_weather"], AGENT_TOOLS)
        assert available_client_side_tools.get() == {"get_weather"}

    def test_non_string_non_dict_silently_skipped(self):
        """Invalid types are skipped, not errored."""
        apply_tool_filter([123, "get_weather"], AGENT_TOOLS)  # type: ignore[list-item]
        assert available_client_side_tools.get() == {"get_weather"}

    def test_duplicate_names_deduplicated(self):
        """Duplicate names just collapse — no error."""
        apply_tool_filter(["get_weather", "get_weather", "show_map"], AGENT_TOOLS)
        assert available_client_side_tools.get() == {"get_weather", "show_map"}


class TestToolNotAvailableEnforcement:
    """Tests that client_side_tool_fn returns error ToolMessage when tool is filtered out."""

    def test_tool_not_in_allowed_set_returns_error(self):
        token = available_client_side_tools.set({"other_tool"})
        try:
            from uipath.agent.models.agent import AgentClientSideToolResourceConfig

            from uipath_langchain.agent.tools.client_side_tool import (
                create_client_side_tool,
            )

            resource = AgentClientSideToolResourceConfig(
                name="my_tool",
                description="A test tool",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                output_schema=None,
            )

            tool = create_client_side_tool(resource)

            assert tool.coroutine is not None
            result = _run_sync(tool.coroutine(tool_call_id="tc-1", query="test"))

            assert result.status == "error"
            assert "not available" in result.content
        finally:
            available_client_side_tools.reset(token)

    def test_tool_in_allowed_set_proceeds(self):
        """When tool IS in the allowed set, it should NOT return an error."""
        token = available_client_side_tools.set({"my_tool"})
        try:
            from unittest.mock import patch

            from uipath.agent.models.agent import AgentClientSideToolResourceConfig

            from uipath_langchain.agent.tools.client_side_tool import (
                create_client_side_tool,
            )

            resource = AgentClientSideToolResourceConfig(
                name="my_tool",
                description="A test tool",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                output_schema=None,
            )

            with (
                patch(
                    "uipath_langchain.agent.tools.client_side_tool.durable_interrupt",
                    side_effect=lambda fn: fn,
                ),
                patch(
                    "uipath_langchain.agent.tools.client_side_tool.mockable",
                    side_effect=lambda **kw: lambda fn: fn,
                ),
            ):
                tool = create_client_side_tool(resource)
                assert tool.coroutine is not None
                result = _run_sync(tool.coroutine(tool_call_id="tc-1", query="test"))
                if hasattr(result, "status"):
                    assert result.status != "error"
        finally:
            available_client_side_tools.reset(token)

    def test_none_allowed_set_permits_all(self):
        """When available_client_side_tools is None (CAS default), all tools proceed."""
        token = available_client_side_tools.set(None)
        try:
            from unittest.mock import patch

            from uipath.agent.models.agent import AgentClientSideToolResourceConfig

            from uipath_langchain.agent.tools.client_side_tool import (
                create_client_side_tool,
            )

            resource = AgentClientSideToolResourceConfig(
                name="any_tool",
                description="A test tool",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
                output_schema=None,
            )

            with (
                patch(
                    "uipath_langchain.agent.tools.client_side_tool.durable_interrupt",
                    side_effect=lambda fn: fn,
                ),
                patch(
                    "uipath_langchain.agent.tools.client_side_tool.mockable",
                    side_effect=lambda **kw: lambda fn: fn,
                ),
            ):
                tool = create_client_side_tool(resource)

                assert tool.coroutine is not None
                result = _run_sync(tool.coroutine(tool_call_id="tc-1", q="test"))
                if hasattr(result, "status"):
                    assert result.status != "error"
        finally:
            available_client_side_tools.reset(token)
