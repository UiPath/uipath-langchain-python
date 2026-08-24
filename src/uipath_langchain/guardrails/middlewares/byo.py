"""Bring Your Own Guardrail (BYOG) middleware."""

import logging
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from uipath.core.guardrails import GuardrailSelector
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    GuardrailScope,
)
from uipath.platform.guardrails.guardrails import (
    BYO_VALIDATOR_TYPE,
    ValidatorParameter,
)

from ..enums import GuardrailExecutionStage
from ..models import GuardrailAction
from ._base import (
    BUILT_IN_VALIDATOR_GUARDRAIL_TYPE,
    BuiltInGuardrailMiddlewareMixin,
)

logger = logging.getLogger(__name__)


class UiPathByoGuardrailMiddleware(BuiltInGuardrailMiddlewareMixin):
    """Middleware for Bring Your Own Guardrail (BYOG) validations.

    BYOG lets an organization plug its own safety validator (e.g. a cloud
    content-safety subscription, a vendor validation service, or a custom
    Integration Service connector) into UiPath guardrails. An admin first
    creates the configuration under ``Admin -> AI Trust Layer -> Guardrails
    Configurations``; this middleware then references it purely by its validator
    name, which is unique per tenant. The Integration Service connection to use
    is resolved server-side from the configuration, so an admin rebind is always
    honored.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from uipath.core.guardrails import GuardrailScope
        from uipath_langchain.guardrails import (
            BlockAction,
            UiPathByoGuardrailMiddleware,
        )

        @tool
        def analyze_joke_syntax(joke: str) -> str:
            \"\"\"Analyze the syntax of a joke.\"\"\"
            return f"Words: {len(joke.split())}"

        # Customer-managed guardrail on agent input/output
        middleware_agent = UiPathByoGuardrailMiddleware(
            validator_name="my-harmful-content-guardrail",
            scopes=[GuardrailScope.AGENT],
            action=BlockAction(),
        )

        # Same configuration applied to a specific tool
        middleware_tool = UiPathByoGuardrailMiddleware(
            validator_name="my-harmful-content-guardrail",
            scopes=[GuardrailScope.TOOL],
            action=BlockAction(),
            tools=[analyze_joke_syntax],
        )

        agent = create_agent(
            model=llm,
            tools=[analyze_joke_syntax],
            middleware=[*middleware_agent, *middleware_tool],
        )
        ```

    Args:
        validator_name: The BYOG configuration's validator name
            (``byoValidatorName``), as shown in Admin -> AI Trust Layer ->
            Guardrails Configurations or by ``uip agent guardrails list --byo``.
            Unique per tenant; the Integration Service connection to use is
            resolved server-side from the configuration.
        scopes: List of scopes where the guardrail applies (Agent, LLM, Tool).
            BYOG validators are not scope-restricted -- all three scopes are
            available, as with the built-in validators.
        action: Action to take when the validation fails (LogAction,
            BlockAction, EscalateAction, or a custom GuardrailAction).
        validator_parameters: Optional list of validator parameters. BYO
            parameter schemas are connector-defined, so values are passed
            through as-is; read the ids and allowed values from the
            validator's ``Parameters`` in ``uip agent guardrails list``.
        tools: Required when TOOL scope is specified. List of tool names or
            tool objects to apply the guardrail to. Must contain at least one
            tool. Can be a mix of strings (tool names) or BaseTool objects.
            If TOOL scope is not specified, this parameter is ignored.
        stage: Optional execution stage controlling when the guardrail runs.
            ``PRE`` evaluates before the target executes (registers only the
            ``before_*`` hook), ``POST`` evaluates after (only the ``after_*``
            hook), and ``PRE_AND_POST`` evaluates both. Applies to all scopes
            (Agent, LLM, Tool). Defaults to ``GuardrailExecutionStage.PRE_AND_POST``.
        name: Optional name for the guardrail (defaults to
            ``BYO Guardrail (<validator_name>)``).
        description: Optional description for the guardrail.
        enabled_for_evals: Whether this guardrail is enabled for evaluation
            scenarios. Defaults to True.
    """

    def __init__(
        self,
        validator_name: str,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        *,
        validator_parameters: Sequence[ValidatorParameter] | None = None,
        tools: Sequence[str | BaseTool] | None = None,
        stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
        name: str | None = None,
        description: str | None = None,
        enabled_for_evals: bool = True,
    ):
        """Initialize Bring Your Own Guardrail middleware."""
        if not validator_name or not validator_name.strip():
            raise ValueError("validator_name must be a non-empty string")
        if not scopes:
            raise ValueError("At least one scope must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not isinstance(enabled_for_evals, bool):
            raise ValueError("enabled_for_evals must be a boolean")

        self._tool_names = self._resolve_tool_names(tools)
        scopes_list = list(scopes)
        self._require_tools_for_tool_scope(scopes_list)

        self.scopes = scopes_list
        self.action = action
        self.validator_name = validator_name
        self.validator_parameters = list(validator_parameters or [])
        self._tool_stage = stage
        self._name = name or f"BYO Guardrail ({validator_name})"
        self.enabled_for_evals = enabled_for_evals
        self._description = (
            description or f"Bring Your Own Guardrail validation '{validator_name}'"
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
        selector_kwargs: dict[str, Any] = {"scopes": self.scopes}
        if GuardrailScope.TOOL in self.scopes:
            selector_kwargs["match_names"] = self._tool_names

        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=self._name,
            description=self._description,
            enabled_for_evals=self.enabled_for_evals,
            selector=GuardrailSelector(**selector_kwargs),
            guardrail_type=BUILT_IN_VALIDATOR_GUARDRAIL_TYPE,
            validator_type=BYO_VALIDATOR_TYPE,
            validator_parameters=self.validator_parameters,
            byo_validator_name=self.validator_name,
        )
