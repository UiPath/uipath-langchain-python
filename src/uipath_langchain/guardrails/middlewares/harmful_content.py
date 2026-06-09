"""Harmful content detection guardrail middleware."""

import logging
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from uipath.core.guardrails import GuardrailSelector
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    EnumListParameterValue,
    GuardrailScope,
    MapEnumParameterValue,
)

from ..enums import GuardrailExecutionStage
from ..models import GuardrailAction, HarmfulContentEntity
from ._base import BuiltInGuardrailMiddlewareMixin
from ._utils import sanitize_tool_name

logger = logging.getLogger(__name__)


class UiPathHarmfulContentMiddleware(BuiltInGuardrailMiddlewareMixin):
    """Middleware for harmful content detection using UiPath guardrails.

    Supports all scopes (AGENT, LLM, TOOL) and both PRE and POST stages.

    Args:
        scopes: List of scopes where the guardrail applies (Agent, LLM, Tool).
        action: Action to take when harmful content is detected.
        entities: List of harmful content entities to detect with their thresholds.
        tools: Required when TOOL scope is specified. List of tool names or tool objects.
        name: Optional name for the guardrail.
        description: Optional description for the guardrail.
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
    """

    def __init__(
        self,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        entities: Sequence[HarmfulContentEntity],
        *,
        tools: Sequence[str | BaseTool] | None = None,
        stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
        name: str = "Harmful Content Detection",
        description: str | None = None,
        enabled_for_evals: bool = True,
    ):
        """Initialize harmful content detection guardrail middleware."""
        if not scopes:
            raise ValueError("At least one scope must be specified")
        if not entities:
            raise ValueError("At least one entity must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not isinstance(enabled_for_evals, bool):
            raise ValueError("enabled_for_evals must be a boolean")

        self._tool_names: list[str] | None = None
        if tools is not None:
            tool_name_list = []
            for tool_or_name in tools:
                if isinstance(tool_or_name, BaseTool):
                    tool_name_list.append(sanitize_tool_name(tool_or_name.name))
                elif isinstance(tool_or_name, str):
                    tool_name_list.append(sanitize_tool_name(tool_or_name))
                else:
                    raise ValueError(
                        f"tools must contain strings or BaseTool objects, got {type(tool_or_name)}"
                    )
            self._tool_names = tool_name_list

        scopes_list = list(scopes)
        if GuardrailScope.TOOL in scopes_list:
            if self._tool_names is None or len(self._tool_names) == 0:
                raise ValueError(
                    "Tool scope is specified but tools is None or empty. "
                    "Tool scope guardrails require at least one tool to be specified."
                )

        self.scopes = scopes_list
        self.action = action
        self.entities = list(entities)
        self._tool_stage = stage
        self._name = name
        self.enabled_for_evals = enabled_for_evals
        self._description = (
            description
            or f"Detects harmful content: {', '.join(e.name for e in entities)}"
        )

        self._guardrail = self._create_guardrail()
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances from decorated functions.

        AGENT/LLM hooks are built by the shared, stage-gated
        ``_build_message_hooks`` helper (``PRE`` → ``before_*`` only, ``POST`` →
        ``after_*`` only, ``PRE_AND_POST`` → both), so a guardrail validates (and
        acts, e.g. escalates) at a single checkpoint instead of twice per run.
        """
        instances: list[AgentMiddleware] = []
        guardrail_name = self._name.replace(" ", "_")

        if GuardrailScope.AGENT in self.scopes:
            instances.extend(
                self._build_message_hooks(
                    GuardrailScope.AGENT, self._tool_stage, guardrail_name
                )
            )
        if GuardrailScope.LLM in self.scopes:
            instances.extend(
                self._build_message_hooks(
                    GuardrailScope.LLM, self._tool_stage, guardrail_name
                )
            )
        if GuardrailScope.TOOL in self.scopes:
            instances.append(self._create_tool_wrap_hook(guardrail_name))

        return instances

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _create_guardrail(self) -> BuiltInValidatorGuardrail:
        """Create BuiltInValidatorGuardrail from configuration."""
        entity_names = [entity.name for entity in self.entities]
        entity_thresholds: dict[str, float] = {
            entity.name: entity.threshold for entity in self.entities
        }

        validator_parameters = [
            EnumListParameterValue(
                parameter_type="enum-list",
                id="harmfulContentEntities",
                value=entity_names,
            ),
            MapEnumParameterValue(
                parameter_type="map-enum",
                id="harmfulContentEntityThresholds",
                value=entity_thresholds,
            ),
        ]

        selector_kwargs: dict[str, Any] = {"scopes": self.scopes}
        if GuardrailScope.TOOL in self.scopes:
            selector_kwargs["match_names"] = self._tool_names

        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=self._name,
            description=self._description,
            enabled_for_evals=self.enabled_for_evals,
            selector=GuardrailSelector(**selector_kwargs),
            guardrail_type="builtInValidator",
            validator_type="harmful_content",
            validator_parameters=validator_parameters,
        )
