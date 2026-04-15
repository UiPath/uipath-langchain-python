"""Prompt injection detection guardrail middleware."""

import logging
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState, before_model
from langgraph.runtime import Runtime
from uipath.core.guardrails import GuardrailSelector
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailScope
from uipath.platform.guardrails.guardrails import NumberParameterValue

from ..models import GuardrailAction
from ._base import BuiltInGuardrailMiddlewareMixin

logger = logging.getLogger(__name__)


class UiPathPromptInjectionMiddleware(BuiltInGuardrailMiddlewareMixin):
    """Middleware for prompt injection detection using UiPath guardrails.

    Example:
        ```python
        from uipath_langchain.guardrails import (
            UiPathPromptInjectionMiddleware,
            LogAction,
            GuardrailScope,
        )
        from uipath_langchain.guardrails.actions import LoggingSeverityLevel

        middleware = UiPathPromptInjectionMiddleware(
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            threshold=0.5,
            enabled_for_evals=True,
        )
        ```

    Args:
        scopes: Optional list of scopes where the guardrail applies. Only LLM scope is
            supported. Defaults to [GuardrailScope.LLM] when not provided.
        action: Action to take when prompt injection is detected (LogAction or BlockAction)
        threshold: Detection threshold (0.0 to 1.0)
        name: Optional name for the guardrail (defaults to "Prompt Injection Detection")
        description: Optional description for the guardrail
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
            Defaults to True.
    """

    def __init__(
        self,
        action: GuardrailAction,
        threshold: float = 0.5,
        *,
        scopes: Sequence[GuardrailScope] | None = None,
        name: str = "Prompt Injection Detection",
        description: str | None = None,
        enabled_for_evals: bool = True,
    ):
        """Initialize prompt injection detection guardrail middleware."""
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        if not isinstance(enabled_for_evals, bool):
            raise ValueError("enabled_for_evals must be a boolean")

        scopes_list = list(scopes) if scopes is not None else [GuardrailScope.LLM]
        if scopes_list != [GuardrailScope.LLM]:
            raise ValueError(
                "Prompt injection detection only supports LLM scope. "
                "Please use scopes=[GuardrailScope.LLM] or omit scopes (defaults to [LLM])."
            )

        self.scopes = [GuardrailScope.LLM]
        self.action = action
        self.threshold = threshold
        self._name = name
        self.enabled_for_evals = enabled_for_evals
        self._description = (
            description
            or f"Detects prompt injection attempts with threshold {threshold}"
        )

        self._guardrail = self._create_guardrail()
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances from decorated functions."""
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
        validator_parameters = [
            NumberParameterValue(
                parameter_type="number",
                id="threshold",
                value=self.threshold,
            ),
        ]

        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=self._name,
            description=self._description,
            enabled_for_evals=self.enabled_for_evals,
            selector=GuardrailSelector(scopes=self.scopes),
            guardrail_type="builtInValidator",
            validator_type="prompt_injection",
            validator_parameters=validator_parameters,
        )
