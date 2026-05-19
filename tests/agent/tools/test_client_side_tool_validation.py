"""Tests for client-side tool validation and filtering logic."""

import pytest

from uipath_langchain.agent.tools.client_side_tool import (
    ClientSideToolInfo,
    available_client_side_tools,
    validate_and_apply_tool_filter,
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


class TestValidateAndApplyToolFilter:
    """Tests for validate_and_apply_tool_filter."""

    def test_valid_declarations_set_filter(self):
        declared = [
            {"name": "get_weather"},
            {"name": "show_map"},
        ]
        validate_and_apply_tool_filter(declared, AGENT_TOOLS)

        allowed = available_client_side_tools.get()
        assert allowed == {"get_weather", "show_map"}

    def test_missing_required_tool_raises(self):
        declared = [{"name": "get_weather"}]  # missing show_map

        with pytest.raises(ValueError, match="Missing required client-side tools"):
            validate_and_apply_tool_filter(declared, AGENT_TOOLS)

    def test_input_schema_mismatch_raises(self):
        declared = [
            {
                "name": "get_weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
            {"name": "show_map"},
        ]

        with pytest.raises(ValueError, match="inputSchema does not match"):
            validate_and_apply_tool_filter(declared, AGENT_TOOLS)

    def test_output_schema_mismatch_raises(self):
        declared = [
            {
                "name": "get_weather",
                "outputSchema": {
                    "type": "object",
                    "properties": {"temperature": {"type": "string"}},
                },
            },
            {"name": "show_map"},
        ]

        with pytest.raises(ValueError, match="outputSchema does not match"):
            validate_and_apply_tool_filter(declared, AGENT_TOOLS)

    def test_unknown_extra_tools_are_ignored(self):
        declared = [
            {"name": "get_weather"},
            {"name": "show_map"},
            {"name": "unknown_tool"},
        ]
        validate_and_apply_tool_filter(declared, AGENT_TOOLS)

        allowed = available_client_side_tools.get()
        assert allowed is not None
        assert "unknown_tool" in allowed
        assert "get_weather" in allowed

    def test_string_declarations_accepted(self):
        declared = ["get_weather", "show_map"]
        validate_and_apply_tool_filter(declared, AGENT_TOOLS)

        allowed = available_client_side_tools.get()
        assert allowed == {"get_weather", "show_map"}

    def test_missing_name_field_raises(self):
        declared = [{"inputSchema": {}}]

        with pytest.raises(ValueError, match="missing required 'name' field"):
            validate_and_apply_tool_filter(declared, AGENT_TOOLS)

    def test_invalid_type_raises(self):
        declared = [123]

        with pytest.raises(ValueError, match="must be a dict or string"):
            validate_and_apply_tool_filter(declared, AGENT_TOOLS)

    def test_duplicate_name_raises(self):
        declared = [
            {"name": "get_weather"},
            {"name": "get_weather"},
            {"name": "show_map"},
        ]

        with pytest.raises(ValueError, match="Duplicate client-side tool"):
            validate_and_apply_tool_filter(declared, AGENT_TOOLS)

    def test_matching_schemas_pass(self):
        declared = [
            {
                "name": "get_weather",
                "inputSchema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
                "outputSchema": {
                    "type": "object",
                    "properties": {"temp": {"type": "number"}},
                },
            },
            {"name": "show_map"},
        ]
        validate_and_apply_tool_filter(declared, AGENT_TOOLS)

        allowed = available_client_side_tools.get()
        assert allowed is not None
        assert "get_weather" in allowed


class TestToolNotAvailableEnforcement:
    """Tests that client_side_tool_fn returns error ToolMessage when tool is filtered out."""

    def test_tool_not_in_allowed_set_returns_error(self):
        token = available_client_side_tools.set({"other_tool"})
        try:
            from unittest.mock import AsyncMock, patch

            from uipath.agent.models.agent import AgentClientSideToolResourceConfig

            resource = AgentClientSideToolResourceConfig(
                name="my_tool",
                description="A test tool",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                output_schema=None,
            )

            from uipath_langchain.agent.tools.client_side_tool import (
                create_client_side_tool,
            )

            tool = create_client_side_tool(resource)

            import asyncio

            result = asyncio.get_event_loop().run_until_complete(
                tool.coroutine(tool_call_id="tc-1", query="test")
            )

            assert result.status == "error"
            assert "not available" in result.content
        finally:
            available_client_side_tools.reset(token)

    def test_tool_in_allowed_set_proceeds(self):
        """When tool IS in the allowed set, it should NOT return an error.

        We can't fully test execution (it would hit durable_interrupt),
        but we verify the availability check passes by patching the interrupt.
        """
        token = available_client_side_tools.set({"my_tool"})
        try:
            from unittest.mock import AsyncMock, patch

            from uipath.agent.models.agent import AgentClientSideToolResourceConfig

            resource = AgentClientSideToolResourceConfig(
                name="my_tool",
                description="A test tool",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                output_schema=None,
            )

            from uipath_langchain.agent.tools.client_side_tool import (
                create_client_side_tool,
            )

            tool = create_client_side_tool(resource)

            import asyncio

            # Patch durable_interrupt to avoid GraphInterrupt
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
                # Re-create tool after patching
                tool = create_client_side_tool(resource)
                result = asyncio.get_event_loop().run_until_complete(
                    tool.coroutine(tool_call_id="tc-1", query="test")
                )
                # Should NOT be an error ToolMessage — it proceeded past the availability check
                if hasattr(result, "status"):
                    assert result.status != "error"
        finally:
            available_client_side_tools.reset(token)

    def test_none_allowed_set_permits_all(self):
        """When available_client_side_tools is None (CAS default), all tools proceed."""
        token = available_client_side_tools.set(None)
        try:
            from uipath.agent.models.agent import AgentClientSideToolResourceConfig

            resource = AgentClientSideToolResourceConfig(
                name="any_tool",
                description="A test tool",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
                output_schema=None,
            )

            from unittest.mock import patch

            from uipath_langchain.agent.tools.client_side_tool import (
                create_client_side_tool,
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

                import asyncio

                result = asyncio.get_event_loop().run_until_complete(
                    tool.coroutine(tool_call_id="tc-1", q="test")
                )
                if hasattr(result, "status"):
                    assert result.status != "error"
        finally:
            available_client_side_tools.reset(token)
