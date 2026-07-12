"""LLM-as-judge guardrail middleware."""

import logging
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from uipath.core.guardrails import GuardrailSelector
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailScope

# Only enum-list / map-enum / number are re-exported from the package __init__;
# the text / enum / text-list value types live in the submodule (same import
# style prompt_injection.py uses for NumberParameterValue).
from uipath.platform.guardrails.guardrails import (
    EnumParameterValue,
    NumberParameterValue,
    TextListParameterValue,
    TextParameterValue,
)

from ..enums import GuardrailExecutionStage
from ..models import GuardrailAction
from ._base import BuiltInGuardrailMiddlewareMixin

logger = logging.getLogger(__name__)

# Threshold scale matches the backend OOTB catalog: any float in [0, 6], default 2
# (the catalog's UI step of 2 is only a slider increment; the server accepts any
# value in range). HIGHER = more lenient (only flag clear violations), LOWER = stricter.
_THRESHOLD_MIN = 0.0
_THRESHOLD_MAX = 6.0
_THRESHOLD_DEFAULT = 2.0

# Input limits mirror the backend OOTB catalog / LlmAsJudgeProviderApi; keep in sync.
_MAX_GUARDRAIL_TEXT_LENGTH = 4000
_MAX_EXAMPLE_LENGTH = 1000
_MAX_EXAMPLES_PER_LIST = 2


class UiPathLLMAsJudgeMiddleware(BuiltInGuardrailMiddlewareMixin):
    """Middleware for LLM-as-judge evaluation using UiPath guardrails.

    The customer expresses a rule in natural language (``guardrail_text``) and
    picks a judge model; a judge LLM decides whether the evaluated payload
    complies. Supports all scopes (AGENT, LLM, TOOL) and both PRE and POST stages.

    Example:
        ```python
        from uipath_langchain.guardrails import (
            UiPathLLMAsJudgeMiddleware,
            LogAction,
            GuardrailScope,
        )
        from uipath_langchain.guardrails.actions import LoggingSeverityLevel

        middleware = UiPathLLMAsJudgeMiddleware(
            scopes=[GuardrailScope.AGENT],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            guardrail_text="The answer must not contain financial advice.",
            model="gpt-4o-2024-08-06",
        )
        ```

    Args:
        scopes: List of scopes where the guardrail applies (Agent, LLM, Tool).
        action: Action to take when the judge flags a violation.
        guardrail_text: The natural-language rule the judge evaluates against
            (at most 4000 characters).
        model: The judge model to use (a model id supported by LLM Gateway).
        positive_examples: Optional example payloads that comply with the rule
            (at most 2 entries, each at most 1000 characters).
        negative_examples: Optional example payloads that violate the rule
            (at most 2 entries, each at most 1000 characters).
        threshold: Strictness on a 0-6 scale (default 2); higher is more lenient.
        tools: Required when TOOL scope is specified. List of tool names or tool objects.
        stage: PRE, POST, or PRE_AND_POST evaluation (defaults to PRE_AND_POST).
        name: Optional name for the guardrail.
        description: Optional description for the guardrail.
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
    """

    def __init__(
        self,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        guardrail_text: str,
        model: str,
        *,
        positive_examples: Sequence[str] | None = None,
        negative_examples: Sequence[str] | None = None,
        threshold: float = _THRESHOLD_DEFAULT,
        tools: Sequence[str | BaseTool] | None = None,
        stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
        name: str = "LLM as Judge",
        description: str | None = None,
        enabled_for_evals: bool = True,
    ):
        """Initialize LLM-as-judge guardrail middleware."""
        if not scopes:
            raise ValueError("At least one scope must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not guardrail_text or not guardrail_text.strip():
            raise ValueError("guardrail_text must be a non-empty string")
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not _THRESHOLD_MIN <= threshold <= _THRESHOLD_MAX:
            raise ValueError(
                f"Threshold must be between {_THRESHOLD_MIN} and {_THRESHOLD_MAX}, "
                f"got {threshold}"
            )
        if not isinstance(enabled_for_evals, bool):
            raise ValueError("enabled_for_evals must be a boolean")
        if len(guardrail_text) > _MAX_GUARDRAIL_TEXT_LENGTH:
            raise ValueError(
                f"guardrail_text exceeds the {_MAX_GUARDRAIL_TEXT_LENGTH}-character "
                f"limit (got {len(guardrail_text)})"
            )
        positive_examples = list(positive_examples or [])
        negative_examples = list(negative_examples or [])
        for label, examples in (
            ("positive_examples", positive_examples),
            ("negative_examples", negative_examples),
        ):
            if len(examples) > _MAX_EXAMPLES_PER_LIST:
                raise ValueError(
                    f"{label} allows at most {_MAX_EXAMPLES_PER_LIST} examples "
                    f"(got {len(examples)})"
                )
            if any(len(e) > _MAX_EXAMPLE_LENGTH for e in examples):
                raise ValueError(
                    f"each {label} entry must be at most {_MAX_EXAMPLE_LENGTH} characters"
                )

        self._tool_names = self._resolve_tool_names(tools)
        scopes_list = list(scopes)
        self._require_tools_for_tool_scope(scopes_list)

        self.scopes = scopes_list
        self.action = action
        self.guardrail_text = guardrail_text
        self.model = model
        self.positive_examples = positive_examples
        self.negative_examples = negative_examples
        self.threshold = threshold
        self._tool_stage = stage
        self._name = name
        self.enabled_for_evals = enabled_for_evals
        self._description = description or "Evaluates content with an LLM-as-judge rule"

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
        validator_parameters: list[Any] = [
            TextParameterValue(
                parameter_type="text",
                id="guardrailText",
                value=self.guardrail_text,
            ),
            EnumParameterValue(
                parameter_type="enum",
                id="model",
                value=self.model,
            ),
            NumberParameterValue(
                parameter_type="number",
                id="threshold",
                value=self.threshold,
            ),
        ]
        if self.positive_examples:
            validator_parameters.append(
                TextListParameterValue(
                    parameter_type="text-list",
                    id="positiveExamples",
                    value=self.positive_examples,
                )
            )
        if self.negative_examples:
            validator_parameters.append(
                TextListParameterValue(
                    parameter_type="text-list",
                    id="negativeExamples",
                    value=self.negative_examples,
                )
            )

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
            validator_type="llm_as_judge",
            validator_parameters=validator_parameters,
        )
