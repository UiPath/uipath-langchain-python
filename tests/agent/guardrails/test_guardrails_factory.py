"""Tests for guardrails_factory.build_guardrails_with_actions."""

import logging
import types
from typing import cast

from uipath.agent.models.agent import (
    AgentGuardrail as AgentGuardrailModel,
)
from uipath.agent.models.agent import (
    AgentGuardrailActionType,
    AgentGuardrailBlockAction,
    AgentGuardrailLogAction,
    AgentGuardrailSeverityLevel,
    AgentGuardrailUnknownAction,
)

from uipath_langchain.agent.guardrails.actions.block_action import BlockAction
from uipath_langchain.agent.guardrails.actions.log_action import LogAction
from uipath_langchain.agent.guardrails.guardrails_factory import (
    build_guardrails_with_actions,
)


class TestGuardrailsFactory:
    def test_none_returns_empty(self) -> None:
        assert build_guardrails_with_actions(None) == []

    def test_empty_list_returns_empty(self) -> None:
        assert build_guardrails_with_actions([]) == []

    def test_block_action_is_mapped_with_reason(self) -> None:
        guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_name",
                action=AgentGuardrailBlockAction(
                    action_type=AgentGuardrailActionType.BLOCK,
                    reason="stop now",
                ),
            ),
        )

        result = build_guardrails_with_actions([guardrail])

        assert len(result) == 1
        gr, action = result[0]
        assert gr is guardrail
        assert isinstance(action, BlockAction)
        assert action.reason == "stop now"

    def test_log_action_is_mapped_with_message_and_severity_level(self) -> None:
        """LOG action is mapped to LogAction with correct message and logging level."""
        guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_log",
                action=AgentGuardrailLogAction(
                    action_type=AgentGuardrailActionType.LOG,
                    message="note this",
                    severity_level=AgentGuardrailSeverityLevel.WARNING,
                ),
            ),
        )

        result = build_guardrails_with_actions([guardrail])

        assert len(result) == 1
        gr, action = result[0]
        assert gr is guardrail
        assert isinstance(action, LogAction)
        assert action.message == "note this"
        assert action.level == logging.WARNING

    def test_unknown_actions_are_ignored(self) -> None:
        """Unknown actions are ignored by the factory."""
        log_guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_1",
                action=AgentGuardrailUnknownAction(
                    action_type=AgentGuardrailActionType.UNKNOWN,
                ),
            ),
        )
        # Mixing UNKNOWN with BLOCK yields only one mapped tuple (BLOCK)
        block_guardrail = cast(
            AgentGuardrailModel,
            types.SimpleNamespace(
                name="guardrail_2",
                action=AgentGuardrailBlockAction(
                    action_type=AgentGuardrailActionType.BLOCK,
                    reason="block it",
                ),
            ),
        )
        result = build_guardrails_with_actions([log_guardrail, block_guardrail])
        assert len(result) == 1
        gr, action = result[0]
        assert gr is block_guardrail
        assert isinstance(action, BlockAction)
