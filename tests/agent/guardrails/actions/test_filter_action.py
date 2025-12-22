"""Tests for FilterAction in isolation.

These tests validate the filter action behavior independently of guardrail rules.
The filter action is responsible for removing specified fields from tool inputs/outputs
when invoked, regardless of which rule triggered the guardrail.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from uipath.core.guardrails.guardrails import FieldReference, FieldSource
from uipath.platform.guardrails import GuardrailScope
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.exceptions import AgentTerminationException
from uipath_langchain.agent.guardrails.actions.filter_action import FilterAction
from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


def create_field_reference(path: str, source: str) -> FieldReference:
    """Create a FieldReference object."""
    return FieldReference(
        path=path,
        source=FieldSource.INPUT if source == "input" else FieldSource.OUTPUT,
    )


class TestFilterAction:
    """Test FilterAction for various scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("scope", "stage", "expected_node_name"),
        [
            (
                GuardrailScope.LLM,
                ExecutionStage.PRE_EXECUTION,
                "llm_pre_execution_my_guardrail_v1_filter",
            ),
            (
                GuardrailScope.LLM,
                ExecutionStage.POST_EXECUTION,
                "llm_post_execution_my_guardrail_v1_filter",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.PRE_EXECUTION,
                "agent_pre_execution_my_guardrail_v1_filter",
            ),
            (
                GuardrailScope.AGENT,
                ExecutionStage.POST_EXECUTION,
                "agent_post_execution_my_guardrail_v1_filter",
            ),
        ],
    )
    async def test_node_name_and_exception_for_unsupported_scopes(
        self, scope: GuardrailScope, stage: ExecutionStage, expected_node_name: str
    ) -> None:
        """AGENT/LLM scopes raise AgentTerminationException and node name is sanitized."""
        action = FilterAction()
        guardrail = MagicMock()
        guardrail.name = "My Guardrail v1"

        node_name, node = action.action_node(
            guardrail=guardrail,
            scope=scope,
            execution_stage=stage,
            guarded_component_name="guarded_node_name",
        )

        assert node_name == expected_node_name

        with pytest.raises(AgentTerminationException) as excinfo:
            await node(AgentGuardrailsGraphState(messages=[]))

        # Validate rich error info
        assert (
            excinfo.value.error_info.code
            == f"Python.{UiPathErrorCode.EXECUTION_ERROR.value}"
        )
        assert excinfo.value.error_info.title == "Guardrail filter action not supported"
        assert (
            excinfo.value.error_info.detail
            == f"FilterAction is not supported for scope [{scope.name}] at this time."
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stage,expected_node_name",
        [
            (
                ExecutionStage.PRE_EXECUTION,
                "tool_pre_execution_test_guardrail_v1_filter",
            ),
            (
                ExecutionStage.POST_EXECUTION,
                "tool_post_execution_test_guardrail_v1_filter",
            ),
        ],
    )
    async def test_tool_scope_node_name(
        self, stage: ExecutionStage, expected_node_name: str
    ) -> None:
        """TOOL scope: node name is sanitized correctly."""
        action = FilterAction(fields=[create_field_reference("test", "input")])
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail v1"

        node_name, _ = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=stage,
            guarded_component_name="test_tool",
        )

        assert node_name == expected_node_name

    @pytest.mark.asyncio
    async def test_filter_input_at_pre_execution(self) -> None:
        """Filter single input field at PRE_EXECUTION."""
        fields = [create_field_reference("sentence", "input")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        # Create state with tool call
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Agent___Sentence_Analyzer",
                    "args": {"sentence": "Test sentence"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        # Verify input field was filtered
        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        assert updated_message.tool_calls[0]["args"] == {}

    @pytest.mark.asyncio
    async def test_filter_output_at_post_execution(self) -> None:
        """Filter single output field at POST_EXECUTION."""
        fields = [create_field_reference("content", "output")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        # Create state with tool message
        tool_message = ToolMessage(
            content='{"content": "Test content", "other": "data"}',
            tool_call_id="call_1",
            name="Agent___Sentence_Analyzer",
        )
        state = AgentGuardrailsGraphState(messages=[tool_message])

        result = await node(state)

        # Verify output field was filtered
        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        parsed_content = json.loads(updated_message.content)
        assert "content" not in parsed_content
        assert parsed_content["other"] == "data"

    @pytest.mark.asyncio
    async def test_filter_both_input_and_output_at_pre(self) -> None:
        """Filter both input and output fields at PRE_EXECUTION (only input filtered)."""
        fields = [
            create_field_reference("sentence", "input"),
            create_field_reference("content", "output"),
        ]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Agent___Sentence_Analyzer",
                    "args": {"sentence": "Test sentence"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        # At PRE_EXECUTION, only input fields should be filtered
        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        assert updated_message.tool_calls[0]["args"] == {}

    @pytest.mark.asyncio
    async def test_filter_both_input_and_output_at_post(self) -> None:
        """Filter multiple input and output fields at POST_EXECUTION (only output filtered).

        This test validates:
        - Multiple input fields specified (but not filtered at POST)
        - Multiple output fields filtered at POST_EXECUTION
        - Unspecified fields remain intact
        """
        fields = [
            create_field_reference("sentence", "input"),
            create_field_reference("param", "input"),
            create_field_reference("content", "output"),
            create_field_reference("metadata", "output"),
        ]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        # Create state with both AIMessage (with tool call) and ToolMessage
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Agent___Sentence_Analyzer",
                    "args": {
                        "sentence": "Test sentence",
                        "param": "value",
                        "keep_input": "this",
                    },
                    "id": "call_1",
                }
            ],
        )
        tool_message = ToolMessage(
            content='{"content": "Test content", "metadata": "info", "keep_output": "this"}',
            tool_call_id="call_1",
            name="Agent___Sentence_Analyzer",
        )
        state = AgentGuardrailsGraphState(messages=[ai_message, tool_message])

        result = await node(state)

        # At POST_EXECUTION, only output fields should be filtered (not input)
        assert isinstance(result, Command)
        assert result.update is not None

        # Check input was NOT filtered (all input args remain)
        updated_ai_message = result.update["messages"][0]
        assert updated_ai_message.tool_calls[0]["args"] == {
            "sentence": "Test sentence",
            "param": "value",
            "keep_input": "this",
        }

        # Check multiple output fields were filtered, but unspecified field remains
        updated_tool_message = result.update["messages"][1]
        parsed_content = json.loads(updated_tool_message.content)
        assert "content" not in parsed_content
        assert "metadata" not in parsed_content
        assert parsed_content["keep_output"] == "this"

    # Edge cases and error handling
    @pytest.mark.asyncio
    async def test_filter_with_no_fields_returns_empty(self) -> None:
        """Filter action with no fields returns empty dict."""
        action = FilterAction(fields=[])
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        state = AgentGuardrailsGraphState(messages=[])
        result = await node(state)

        assert result == {}

    @pytest.mark.asyncio
    async def test_filter_input_with_no_matching_tool_call(self) -> None:
        """Filter input when no matching tool call exists returns empty dict."""
        fields = [create_field_reference("sentence", "input")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        # Tool call with different name
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Different_Tool",
                    "args": {"sentence": "Test"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        # No matching tool, should return empty
        assert result == {}

    @pytest.mark.asyncio
    async def test_filter_output_with_no_matching_tool_message(self) -> None:
        """Filter output processes last ToolMessage regardless of name."""
        fields = [create_field_reference("content", "output")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        # Tool message with different name - still gets processed
        tool_message = ToolMessage(
            content='{"content": "Test", "other": "data"}',
            tool_call_id="call_1",
            name="Different_Tool",
        )
        state = AgentGuardrailsGraphState(messages=[tool_message])

        result = await node(state)

        # Output filtering processes last ToolMessage regardless of name
        assert isinstance(result, Command)
        assert result.update is not None
        updated_message = result.update["messages"][0]
        parsed_content = json.loads(updated_message.content)
        assert "content" not in parsed_content
        assert parsed_content["other"] == "data"

    @pytest.mark.asyncio
    async def test_filter_output_with_non_json_content(self) -> None:
        """Filter output with non-JSON content returns empty dict."""
        fields = [create_field_reference("content", "output")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        tool_message = ToolMessage(
            content="Not JSON content",
            tool_call_id="call_1",
            name="Agent___Sentence_Analyzer",
        )
        state = AgentGuardrailsGraphState(messages=[tool_message])

        result = await node(state)

        # Non-JSON content, should return empty
        assert result == {}

    @pytest.mark.asyncio
    async def test_filter_input_field_not_in_args(self) -> None:
        """Filter input when field doesn't exist in args returns empty dict."""
        fields = [create_field_reference("nonexistent_field", "input")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "Agent___Sentence_Analyzer",
                    "args": {"sentence": "Test"},
                    "id": "call_1",
                }
            ],
        )
        state = AgentGuardrailsGraphState(messages=[ai_message])

        result = await node(state)

        # Field doesn't exist, should return empty (no modification)
        assert result == {}

    @pytest.mark.asyncio
    async def test_filter_output_field_not_in_content(self) -> None:
        """Filter output when field doesn't exist in content returns empty dict."""
        fields = [create_field_reference("nonexistent_field", "output")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name="Agent___Sentence_Analyzer",
        )

        tool_message = ToolMessage(
            content='{"content": "Test"}',
            tool_call_id="call_1",
            name="Agent___Sentence_Analyzer",
        )
        state = AgentGuardrailsGraphState(messages=[tool_message])

        result = await node(state)

        # Field doesn't exist, should return empty (no modification)
        assert result == {}

    @pytest.mark.asyncio
    async def test_filter_with_empty_messages_returns_empty(self) -> None:
        """Filter action with empty messages returns empty dict."""
        fields = [create_field_reference("sentence", "input")]
        action = FilterAction(fields=fields)
        guardrail = MagicMock()
        guardrail.name = "Test Guardrail"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="test_tool",
        )

        state = AgentGuardrailsGraphState(messages=[])
        result = await node(state)

        assert result == {}
