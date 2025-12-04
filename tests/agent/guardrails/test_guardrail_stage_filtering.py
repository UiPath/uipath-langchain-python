"""Tests for internal guardrail stage filtering logic."""

from unittest.mock import MagicMock

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
