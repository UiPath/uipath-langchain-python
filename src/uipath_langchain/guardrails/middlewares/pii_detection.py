"""PII detection guardrail middleware."""

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
from ..models import GuardrailAction, PIIDetectionEntity
from ._base import BuiltInGuardrailMiddlewareMixin

logger = logging.getLogger(__name__)


class UiPathPIIDetectionMiddleware(BuiltInGuardrailMiddlewareMixin):
    """Middleware for PII detection using UiPath guardrails.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from uipath_langchain.guardrails import (
            UiPathPIIDetectionMiddleware,
            PIIDetectionEntity,
            PIIDetectionEntityType,
            LogAction,
            GuardrailScope,
        )
        from uipath_langchain.guardrails.actions import LoggingSeverityLevel

        @tool
        def analyze_joke_syntax(joke: str) -> str:
            \"\"\"Analyze the syntax of a joke.\"\"\"
            return f"Words: {len(joke.split())}"

        # PII detection for Agent and LLM scopes
        middleware_agent_llm = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[
                PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.ADDRESS, 0.7),
            ],
            enabled_for_evals=True,
        )

        # PII detection for specific tools (using tool reference directly)
        middleware_tool = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.TOOL],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
            tools=[analyze_joke_syntax],
            enabled_for_evals=False,
        )

        agent = create_agent(
            model=llm,
            tools=[analyze_joke_syntax],
            middleware=[*middleware_agent_llm, *middleware_tool],
        )
        ```

    Args:
        scopes: List of scopes where the guardrail applies (Agent, LLM, Tool)
        action: Action to take when PII is detected (LogAction or BlockAction)
        entities: List of PII entities to detect with their thresholds
        tools: Required when TOOL scope is specified. List of tool names or tool objects
            to apply guardrail to. Must contain at least one tool.
            Can be a mix of strings (tool names) or BaseTool objects.
            If TOOL scope is not specified, this parameter is ignored.
        stage: Optional execution stage controlling when the guardrail runs.
            ``PRE`` evaluates before the target executes (registers only the
            ``before_*`` hook), ``POST`` evaluates after (only the ``after_*``
            hook), and ``PRE_AND_POST`` evaluates both. Applies to all scopes
            (Agent, LLM, Tool). Defaults to ``GuardrailExecutionStage.PRE_AND_POST``.
        name: Optional name for the guardrail (defaults to "PII Detection")
        description: Optional description for the guardrail
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
            Defaults to True.
    """

    def __init__(
        self,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        entities: Sequence[PIIDetectionEntity],
        *,
        tools: Sequence[str | BaseTool] | None = None,
        stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
        name: str = "PII Detection",
        description: str | None = None,
        enabled_for_evals: bool = True,
    ):
        """Initialize PII detection guardrail middleware."""
        if not scopes:
            raise ValueError("At least one scope must be specified")
        if not entities:
            raise ValueError("At least one entity must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not isinstance(enabled_for_evals, bool):
            raise ValueError("enabled_for_evals must be a boolean")

        self._tool_names = self._resolve_tool_names(tools)
        scopes_list = list(scopes)
        self._require_tools_for_tool_scope(scopes_list)

        self.scopes = scopes_list
        self.action = action
        self.entities = list(entities)
        self._tool_stage = stage
        self._name = name
        self.enabled_for_evals = enabled_for_evals
        self._description = (
            description
            or f"Detects PII entities: {', '.join(e.name for e in entities)}"
        )

        self._guardrail = self._create_guardrail()
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create scope-gated middleware instances (see ``_build_scope_instances``)."""
        return self._build_scope_instances(self._name.replace(" ", "_"))

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _create_guardrail(self) -> BuiltInValidatorGuardrail:
        """Create BuiltInValidatorGuardrail from configuration."""
        entity_names = [entity.name for entity in self.entities]
        entity_thresholds = {entity.name: entity.threshold for entity in self.entities}

        validator_parameters = [
            EnumListParameterValue(
                parameter_type="enum-list",
                id="entities",
                value=entity_names,
            ),
            MapEnumParameterValue(
                parameter_type="map-enum",
                id="entityThresholds",
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
            validator_type="pii_detection",
            validator_parameters=validator_parameters,
        )
