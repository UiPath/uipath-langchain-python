"""Tests for internal guardrail stage and eval filtering logic."""

from unittest.mock import MagicMock

from uipath.eval._execution_context import execution_id_context
from uipath.platform.guardrails import BuiltInValidatorGuardrail, DeterministicGuardrail

import uipath_langchain.agent.react.guardrails.guardrails_subgraph
from uipath_langchain.agent.guardrails.actions.base_action import GuardrailAction
from uipath_langchain.agent.guardrails.types import ExecutionStage


class TestGuardrailStageFiltering:
    """Test internal stage logic for guardrails."""

    def test_filter_by_stage_prompt_injection(self):
        """Prompt injection guardrails should be skipped in POST_EXECUTION."""
        # 1. Prompt Injection Guardrail (BuiltInValidatorGuardrail)
        pi_guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        pi_guardrail.validator_type = "prompt_injection"
        pi_guardrail.name = "Prompt-Injection"

        # 2. Generic Guardrail (BaseGuardrail)
        generic_guardrail = MagicMock(spec=DeterministicGuardrail)
        generic_guardrail.name = "Generic"

        # 3. Other BuiltIn Guardrail
        other_builtin = MagicMock(spec=BuiltInValidatorGuardrail)
        other_builtin.validator_type = "pii_detection"
        other_builtin.name = "PII"

        action = MagicMock(spec=GuardrailAction)

        # Setup guardrails list
        guardrails = [
            (pi_guardrail, action),
            (generic_guardrail, action),
            (other_builtin, action),
        ]

        # --- PRE_EXECUTION ---
        # Should get ALL guardrails
        pre_filtered = uipath_langchain.agent.react.guardrails.guardrails_subgraph._filter_guardrails_by_stage(
            guardrails, ExecutionStage.PRE_EXECUTION
        )
        assert len(pre_filtered) == 3
        assert pre_filtered[0][0] == pi_guardrail
        assert pre_filtered[1][0] == generic_guardrail
        assert pre_filtered[2][0] == other_builtin

        # --- POST_EXECUTION ---
        # Should SKIP Prompt Injection
        post_filtered = uipath_langchain.agent.react.guardrails.guardrails_subgraph._filter_guardrails_by_stage(
            guardrails, ExecutionStage.POST_EXECUTION
        )
        assert len(post_filtered) == 2
        # Custom should be there
        assert post_filtered[0][0] == generic_guardrail
        # Other builtin should be there
        assert post_filtered[1][0] == other_builtin


class TestGuardrailEvalFiltering:
    """Test filtering of guardrails disabled for evaluations."""

    def _make_guardrail(self, name: str, enabled_for_evals: bool):
        g = MagicMock(spec=BuiltInValidatorGuardrail)
        g.name = name
        g.enabled_for_evals = enabled_for_evals
        return g

    def test_filter_removes_disabled_guardrails_during_eval(self):
        """Guardrails with enabled_for_evals=False are excluded in eval context."""
        action = MagicMock(spec=GuardrailAction)
        enabled = self._make_guardrail("Enabled", enabled_for_evals=True)
        disabled = self._make_guardrail("Disabled", enabled_for_evals=False)

        guardrails = [(enabled, action), (disabled, action)]

        token = execution_id_context.set("eval-123")
        try:
            result = uipath_langchain.agent.react.guardrails.guardrails_subgraph._filter_guardrails_for_evals(
                guardrails
            )
        finally:
            execution_id_context.reset(token)

        assert len(result) == 1
        assert result[0][0] == enabled

    def test_filter_keeps_all_guardrails_outside_eval(self):
        """All guardrails are kept when not running inside an evaluation."""
        action = MagicMock(spec=GuardrailAction)
        enabled = self._make_guardrail("Enabled", enabled_for_evals=True)
        disabled = self._make_guardrail("Disabled", enabled_for_evals=False)

        guardrails = [(enabled, action), (disabled, action)]

        # Ensure we are NOT in eval context
        assert execution_id_context.get() is None

        result = uipath_langchain.agent.react.guardrails.guardrails_subgraph._filter_guardrails_for_evals(
            guardrails
        )

        assert len(result) == 2

    def test_filter_removes_all_when_all_disabled_during_eval(self):
        """All guardrails filtered out when all have enabled_for_evals=False in eval."""
        action = MagicMock(spec=GuardrailAction)
        g1 = self._make_guardrail("G1", enabled_for_evals=False)
        g2 = self._make_guardrail("G2", enabled_for_evals=False)

        guardrails = [(g1, action), (g2, action)]

        token = execution_id_context.set("eval-456")
        try:
            result = uipath_langchain.agent.react.guardrails.guardrails_subgraph._filter_guardrails_for_evals(
                guardrails
            )
        finally:
            execution_id_context.reset(token)

        assert len(result) == 0

    def test_filter_handles_none_input(self):
        """Passing None returns an empty list."""
        result = uipath_langchain.agent.react.guardrails.guardrails_subgraph._filter_guardrails_for_evals(
            None
        )
        assert result == []

    def test_filter_handles_empty_input(self):
        """Passing an empty sequence returns an empty list."""
        result = uipath_langchain.agent.react.guardrails.guardrails_subgraph._filter_guardrails_for_evals(
            []
        )
        assert result == []
