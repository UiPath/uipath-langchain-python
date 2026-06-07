"""Intellectual property detection guardrail middleware."""

import logging
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    after_agent,
    after_model,
)
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from uipath.core.guardrails import GuardrailSelector
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    EnumListParameterValue,
    GuardrailScope,
)

from ..models import GuardrailAction
from ._base import BuiltInGuardrailMiddlewareMixin

logger = logging.getLogger(__name__)


class UiPathIntellectualPropertyMiddleware(BuiltInGuardrailMiddlewareMixin):
    """Middleware for intellectual property detection using UiPath guardrails.

    Supports AGENT and LLM scopes only (not TOOL). POST stage only — registers
    only ``after_agent`` and ``after_model`` hooks.

    Args:
        scopes: List of scopes where the guardrail applies (LLM, AGENT only).
        action: Action to take when IP violation is detected.
        entities: List of IP entity type strings (e.g. ``IntellectualPropertyEntityType.TEXT``).
        name: Optional name for the guardrail.
        description: Optional description for the guardrail.
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
    """

    def __init__(
        self,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        entities: Sequence[str],
        *,
        name: str = "Intellectual Property Detection",
        description: str | None = None,
        enabled_for_evals: bool = True,
    ):
        """Initialize intellectual property detection guardrail middleware."""
        if not scopes:
            raise ValueError("At least one scope must be specified")
        if not entities:
            raise ValueError("At least one entity must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not isinstance(enabled_for_evals, bool):
            raise ValueError("enabled_for_evals must be a boolean")

        scopes_list = list(scopes)
        if GuardrailScope.TOOL in scopes_list:
            raise ValueError(
                "Intellectual property detection does not support TOOL scope. "
                "Please use scopes with AGENT and/or LLM only."
            )

        self.scopes = scopes_list
        self.action = action
        self.entities = list(entities)
        self._name = name
        self.enabled_for_evals = enabled_for_evals
        self._description = (
            description or f"Detects intellectual property: {', '.join(entities)}"
        )

        self._guardrail = self._create_guardrail()
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances — POST only (after_agent, after_model)."""
        instances = []
        middleware_instance = self
        guardrail_name = self._name.replace(" ", "_")

        if GuardrailScope.AGENT in self.scopes:

            async def _after_agent_func(
                state: AgentState[Any], runtime: Runtime
            ) -> None:
                messages = state.get("messages", [])
                middleware_instance._check_messages(list(messages))

            _after_agent_func.__name__ = f"{guardrail_name}_after_agent"
            _after_agent = after_agent(_after_agent_func)
            instances.append(_after_agent)

        if GuardrailScope.LLM in self.scopes:

            async def _after_model_func(
                state: AgentState[Any], runtime: Runtime
            ) -> None:
                messages = state.get("messages", [])
                ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
                if ai_messages:
                    middleware_instance._check_messages([ai_messages[-1]])

            _after_model_func.__name__ = f"{guardrail_name}_after_model"
            _after_model = after_model(_after_model_func)
            instances.append(_after_model)

        return instances

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _create_guardrail(self) -> BuiltInValidatorGuardrail:
        """Create BuiltInValidatorGuardrail from configuration."""
        validator_parameters = [
            EnumListParameterValue(
                parameter_type="enum-list",
                id="ipEntities",
                value=self.entities,
            ),
        ]

        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=self._name,
            description=self._description,
            enabled_for_evals=self.enabled_for_evals,
            selector=GuardrailSelector(scopes=self.scopes),
            guardrail_type="builtInValidator",
            validator_type="intellectual_property",
            validator_parameters=validator_parameters,
        )
