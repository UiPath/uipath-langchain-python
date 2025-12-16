"""Tests for FilterAction guardrail failure behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
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


class TestFilterAction:
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
        "stage",
        [ExecutionStage.PRE_EXECUTION, ExecutionStage.POST_EXECUTION],
    )
    async def test_tool_scope_returns_empty_dict(self, stage: ExecutionStage) -> None:
        """TOOL scope currently performs no-op and returns empty dict."""
        action = FilterAction()
        guardrail = MagicMock()
        guardrail.name = "My Guardrail v1"

        _, node = action.action_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=stage,
            guarded_component_name="test_tool",
        )

        result = await node(AgentGuardrailsGraphState(messages=[]))
        assert result == {}
