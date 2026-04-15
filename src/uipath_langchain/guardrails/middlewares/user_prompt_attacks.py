"""User prompt attacks detection guardrail middleware."""

import logging
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState, before_model
from langgraph.runtime import Runtime
from uipath.core.guardrails import GuardrailSelector
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailScope

from ..models import GuardrailAction
from ._base import BuiltInGuardrailMiddlewareMixin

logger = logging.getLogger(__name__)


class UiPathUserPromptAttacksMiddleware(BuiltInGuardrailMiddlewareMixin):
    """Middleware for user prompt attacks detection using UiPath guardrails.

    Supports LLM scope only. PRE stage only — registers only ``before_model`` hook.
    Takes no entity or threshold parameters.

    Args:
        scopes: Optional list of scopes. Only LLM scope is supported.
            Defaults to [GuardrailScope.LLM].
        action: Action to take when a prompt attack is detected.
        name: Optional name for the guardrail.
        description: Optional description for the guardrail.
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
    """

    def __init__(
        self,
        action: GuardrailAction,
        *,
        scopes: Sequence[GuardrailScope] | None = None,
        name: str = "User Prompt Attacks Detection",
        description: str | None = None,
        enabled_for_evals: bool = True,
    ):
        """Initialize user prompt attacks detection guardrail middleware."""
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not isinstance(enabled_for_evals, bool):
            raise ValueError("enabled_for_evals must be a boolean")

        scopes_list = list(scopes) if scopes is not None else [GuardrailScope.LLM]
        if scopes_list != [GuardrailScope.LLM]:
            raise ValueError(
                "User prompt attacks detection only supports LLM scope. "
                "Please use scopes=[GuardrailScope.LLM] or omit scopes (defaults to [LLM])."
            )

        self.scopes = [GuardrailScope.LLM]
        self.action = action
        self._name = name
        self.enabled_for_evals = enabled_for_evals
        self._description = description or "Detects user prompt attacks"

        self._guardrail = self._create_guardrail()
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances — PRE only (before_model)."""
        instances = []
        middleware_instance = self
        guardrail_name = self._name.replace(" ", "_")

        async def _before_model_func(state: AgentState[Any], runtime: Runtime) -> None:
            messages = state.get("messages", [])
            middleware_instance._check_messages(list(messages))

        _before_model_func.__name__ = f"{guardrail_name}_before_model"
        _before_model = before_model(_before_model_func)
        instances.append(_before_model)

        return instances

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _create_guardrail(self) -> BuiltInValidatorGuardrail:
        """Create BuiltInValidatorGuardrail from configuration."""
        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=self._name,
            description=self._description,
            enabled_for_evals=self.enabled_for_evals,
            selector=GuardrailSelector(scopes=self.scopes),
            guardrail_type="builtInValidator",
            validator_type="user_prompt_attacks",
            validator_parameters=[],
        )
