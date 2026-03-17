"""Deterministic guardrail decorator (tool-level, local rules, no UiPath API call)."""

import inspect
from typing import Any, Callable, Sequence, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from uipath.core.guardrails import GuardrailScope, GuardrailValidationResultType

from ..enums import GuardrailExecutionStage
from ..middlewares._utils import create_modified_tool_result
from ..models import GuardrailAction
from ._base import (
    GuardrailMetadata,
    _evaluate_rules,
    _extract_input,
    _extract_output,
    _rewrap_input,
    _store_guardrail_metadata,
)


def _wrap_tool_with_deterministic_guardrail(
    tool: BaseTool, metadata: GuardrailMetadata
) -> BaseTool:
    """Wrap a BaseTool to apply deterministic rule evaluation.

    Runs local rule functions against input/output dicts, controlled by
    metadata.config["stage"] (default: PRE_AND_POST).

    Args:
        tool: BaseTool instance to wrap.
        metadata: GuardrailMetadata with rules, action, and optional stage.

    Returns:
        The same tool with wrapped invoke/ainvoke.
    """
    action = metadata.config["action"]
    guardrail_name = metadata.name
    stage = metadata.config.get("stage", GuardrailExecutionStage.PRE_AND_POST)
    rules: Sequence[Callable[..., bool]] = metadata.config.get("rules", [])

    def _apply_pre(tool_input: Any) -> Any:
        input_data = _extract_input(tool_input)
        result = _evaluate_rules(
            rules, GuardrailExecutionStage.PRE, input_data, None, guardrail_name
        )
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            modified = action.handle_validation_result(
                result, input_data, guardrail_name
            )
            if modified is not None and isinstance(modified, dict):
                return _rewrap_input(tool_input, modified)
        return tool_input

    def _apply_post(tool_input: Any, result: Any) -> Any:
        input_data = _extract_input(tool_input)
        output_data = _extract_output(result)
        eval_result = _evaluate_rules(
            rules, GuardrailExecutionStage.POST, input_data, output_data, guardrail_name
        )
        if eval_result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            modified = action.handle_validation_result(
                eval_result, output_data, guardrail_name
            )
            if modified is not None:
                if isinstance(result, (ToolMessage, Command)):
                    return create_modified_tool_result(result, modified)
                return modified
        return result

    # BaseTool subclasses are Pydantic models; setattr on methods is blocked.
    # Use __class__ swapping so all Pydantic fields and the StructuredTool interface
    # are fully inherited.
    ConcreteToolType = type(tool)

    class _GuardedTool(ConcreteToolType):  # type: ignore[valid-type, misc]
        def invoke(self, tool_input: Any, config: Any = None, **kwargs: Any) -> Any:
            guarded_input = tool_input
            if _stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                guarded_input = _apply_pre(tool_input)
            result = super().invoke(guarded_input, config, **kwargs)
            if _stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                result = _apply_post(guarded_input, result)
            return result

    # Close over the stage so the subclass methods use the correct value
    _stage = stage

    tool.__class__ = _GuardedTool
    return tool


# Re-export RuleFunction so callers can import from here or from decorators/__init__
RuleFunction = (
    Callable[[dict[str, Any]], bool] | Callable[[dict[str, Any], dict[str, Any]], bool]
)


def _apply_deterministic_guardrail(
    obj: BaseTool,
    rules: Sequence[RuleFunction],
    action: GuardrailAction,
    stage: GuardrailExecutionStage,
    name: str,
    description: str | None,
    enabled_for_evals: bool,
) -> BaseTool:
    """Apply deterministic guardrail to a BaseTool instance."""
    if not isinstance(obj, BaseTool):
        raise ValueError(
            f"@deterministic_guardrail can only be applied to BaseTool instances, "
            f"got {type(obj)}."
        )
    if action is None:
        raise ValueError("action must be provided")
    if not isinstance(action, GuardrailAction):
        raise ValueError("action must be an instance of GuardrailAction")
    if not isinstance(stage, GuardrailExecutionStage):
        raise ValueError(
            f"stage must be a GuardrailExecutionStage instance, got {type(stage)}"
        )
    if not isinstance(enabled_for_evals, bool):
        raise ValueError("enabled_for_evals must be a boolean")

    for i, rule in enumerate(rules):
        if not callable(rule):
            raise ValueError(f"Rule {i + 1} must be callable, got {type(rule)}")
        sig = inspect.signature(rule)
        param_count = len(sig.parameters)
        if param_count not in (1, 2):
            raise ValueError(
                f"Rule {i + 1} must have 1 or 2 parameters, got {param_count}"
            )

    metadata = GuardrailMetadata(
        guardrail_type="deterministic",
        scope=GuardrailScope.TOOL,
        config={
            "rules": list(rules),
            "action": action,
            "stage": stage,
            "enabled_for_evals": enabled_for_evals,
        },
        name=name,
        description=description or "Deterministic guardrail with custom rules",
        guardrail=None,  # No API call — purely local evaluation
        wrap_tool=_wrap_tool_with_deterministic_guardrail,
    )

    _store_guardrail_metadata(obj, metadata)
    return _wrap_tool_with_deterministic_guardrail(obj, metadata)


def deterministic_guardrail(
    func: BaseTool | None = None,
    *,
    rules: Sequence[RuleFunction] = (),
    action: GuardrailAction | None = None,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
    name: str = "Deterministic Guardrail",
    description: str | None = None,
    enabled_for_evals: bool = True,
):
    """Decorator for deterministic guardrails on tools.

    Applies local rule functions to tool inputs and/or outputs — no UiPath API call.
    Scope is always ``GuardrailScope.TOOL``; applying this decorator to anything other
    than a ``BaseTool`` raises ``ValueError``.

    Rule semantics (identical to ``UiPathDeterministicGuardrailMiddleware``):
    - A rule with 1 parameter receives the tool input dict.
    - A rule with 2 parameters receives ``(input_dict, output_dict)``.
    - A rule returns ``True`` to signal a violation, ``False`` to pass.
    - All rules must detect a violation for the guardrail to trigger.
      If any rule passes (returns ``False``), the guardrail passes.
    - Empty ``rules`` always triggers the action (useful for unconditional transforms).

    Args:
        func: Tool to decorate (when used without parentheses).
        rules: Callable rule functions (1 or 2 parameters each).
        action: Action to execute on violation (required).
        stage: When to evaluate — PRE, POST, or PRE_AND_POST (default).
        name: Name for the guardrail.
        description: Optional description.
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
            Defaults to True.

    Example::

        @deterministic_guardrail(
            rules=[lambda args: "donkey" in args.get("joke", "").lower()],
            action=BlockAction(),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Validator",
        )
        @tool
        def analyze_joke_syntax(joke: str) -> str:
            return f"Words: {len(joke.split())}"

    Returns:
        The tool with guardrail-wrapped invoke/ainvoke.
    """

    def _apply(obj: BaseTool) -> BaseTool:
        return _apply_deterministic_guardrail(
            obj,
            rules,
            cast(GuardrailAction, action),
            stage,
            name,
            description,
            enabled_for_evals,
        )

    if func is None:
        # Called as @deterministic_guardrail(...) — return decorator
        return _apply
    else:
        # Called as @deterministic_guardrail (bare, no parentheses)
        if action is None:
            raise ValueError(
                "When using @deterministic_guardrail without parentheses, "
                "you must provide action as a keyword argument."
            )
        return _apply(func)
