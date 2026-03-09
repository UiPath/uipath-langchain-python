"""Tests for voice service — extract_tool_result and resource filtering."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentEscalationResourceConfig,
    AgentIntegrationToolResourceConfig,
    AgentIxpVsEscalationResourceConfig,
    AgentProcessToolResourceConfig,
)
from uipath.runtime import UiPathRuntimeResult, UiPathRuntimeStatus

from uipath_agents.voice.job_runtime import (
    UnsupportedVoiceToolError,
    _filter_voice_resources,
    extract_tool_result,
)


class TestExtractToolResult:
    """Test extraction of tool results from UiPathRuntimeResult."""

    def test_faulted_with_error_dict(self) -> None:
        result = UiPathRuntimeResult(
            status=UiPathRuntimeStatus.FAULTED,
            output={"error": "Unknown tool: foo"},
        )
        text, is_error = extract_tool_result(result)
        assert text == "Unknown tool: foo"
        assert is_error is True

    def test_faulted_with_none_output(self) -> None:
        result = UiPathRuntimeResult(
            status=UiPathRuntimeStatus.FAULTED,
            output=None,
        )
        text, is_error = extract_tool_result(result)
        assert text == "Unknown error"
        assert is_error is True

    def test_faulted_with_string_output(self) -> None:
        result = UiPathRuntimeResult(
            status=UiPathRuntimeStatus.FAULTED,
            output="some error",
        )
        text, is_error = extract_tool_result(result)
        assert text == "some error"
        assert is_error is True

    def test_successful_with_tool_message(self) -> None:
        result = UiPathRuntimeResult(
            status=UiPathRuntimeStatus.SUCCESSFUL,
            output={
                "messages": [
                    {"type": "ai", "content": ""},
                    {
                        "type": "tool",
                        "content": "Sunny, 72F",
                        "name": "weather",
                        "tool_call_id": "c1",
                    },
                ]
            },
        )
        text, is_error = extract_tool_result(result)
        assert text == "Sunny, 72F"
        assert is_error is False

    def test_successful_with_error_tool_message(self) -> None:
        result = UiPathRuntimeResult(
            status=UiPathRuntimeStatus.SUCCESSFUL,
            output={
                "messages": [
                    {"type": "ai", "content": ""},
                    {
                        "type": "tool",
                        "content": "Connection timeout",
                        "status": "error",
                        "name": "api",
                    },
                ]
            },
        )
        text, is_error = extract_tool_result(result)
        assert text == "Connection timeout"
        assert is_error is True

    def test_successful_with_no_messages(self) -> None:
        result = UiPathRuntimeResult(
            status=UiPathRuntimeStatus.SUCCESSFUL,
            output={"messages": []},
        )
        text, is_error = extract_tool_result(result)
        assert text == "{'messages': []}"
        assert is_error is False

    def test_successful_picks_last_tool_message(self) -> None:
        result = UiPathRuntimeResult(
            status=UiPathRuntimeStatus.SUCCESSFUL,
            output={
                "messages": [
                    {"type": "ai", "content": ""},
                    {"type": "tool", "content": "first", "name": "t1"},
                    {"type": "tool", "content": "second", "name": "t2"},
                ]
            },
        )
        text, is_error = extract_tool_result(result)
        assert text == "second"
        assert is_error is False


class TestFilterVoiceResources:
    @staticmethod
    def _make_agent_def(resources: list[Any]) -> MagicMock:
        agent_def = MagicMock()
        agent_def.resources = resources
        return agent_def

    @staticmethod
    def _make_resource(
        spec_class: type, *, is_enabled: bool = True, **attrs: object
    ) -> MagicMock:
        """Create a mock that passes isinstance() checks via __class__ override."""
        resource = MagicMock()
        resource.__class__ = spec_class
        resource.is_enabled = is_enabled
        resource.name = "test_resource"
        for k, v in attrs.items():
            setattr(resource, k, v)
        return resource

    def test_includes_integration_tool(self) -> None:
        resource = self._make_resource(AgentIntegrationToolResourceConfig)
        result = _filter_voice_resources(self._make_agent_def([resource]))
        assert len(result) == 1

    def test_includes_process_tool(self) -> None:
        resource = self._make_resource(AgentProcessToolResourceConfig)
        result = _filter_voice_resources(self._make_agent_def([resource]))
        assert len(result) == 1

    def test_includes_context_tool_semantic(self) -> None:
        settings = MagicMock()
        settings.retrieval_mode = AgentContextRetrievalMode.SEMANTIC
        resource = self._make_resource(AgentContextResourceConfig, settings=settings)
        result = _filter_voice_resources(self._make_agent_def([resource]))
        assert len(result) == 1

    def test_excludes_context_tool_deep_rag(self) -> None:
        settings = MagicMock()
        settings.retrieval_mode = AgentContextRetrievalMode.DEEP_RAG
        resource = self._make_resource(AgentContextResourceConfig, settings=settings)
        with pytest.raises(UnsupportedVoiceToolError):
            _filter_voice_resources(self._make_agent_def([resource]))

    def test_excludes_context_tool_batch_transform(self) -> None:
        settings = MagicMock()
        settings.retrieval_mode = AgentContextRetrievalMode.BATCH_TRANSFORM
        resource = self._make_resource(AgentContextResourceConfig, settings=settings)
        with pytest.raises(UnsupportedVoiceToolError):
            _filter_voice_resources(self._make_agent_def([resource]))

    def test_excludes_disabled_resources(self) -> None:
        resource = self._make_resource(
            AgentIntegrationToolResourceConfig, is_enabled=False
        )
        result = _filter_voice_resources(self._make_agent_def([resource]))
        assert len(result) == 0

    def test_includes_escalation_tool(self) -> None:
        resource = self._make_resource(AgentEscalationResourceConfig)
        result = _filter_voice_resources(self._make_agent_def([resource]))
        assert len(result) == 1

    def test_includes_ixp_vs_escalation_tool(self) -> None:
        resource = self._make_resource(AgentIxpVsEscalationResourceConfig)
        result = _filter_voice_resources(self._make_agent_def([resource]))
        assert len(result) == 1

    def test_excludes_unsupported_type(self) -> None:
        resource = MagicMock()
        resource.is_enabled = True
        resource.name = "unsupported"
        with pytest.raises(UnsupportedVoiceToolError):
            _filter_voice_resources(self._make_agent_def([resource]))
