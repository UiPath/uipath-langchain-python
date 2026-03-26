"""Prompt injection detection guardrail decorator."""

from typing import Any, Callable, cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from uipath.core.guardrails import GuardrailScope, GuardrailSelector
from uipath.platform import UiPath
from uipath.platform.guardrails import BuiltInValidatorGuardrail
from uipath.platform.guardrails.guardrails import NumberParameterValue

from ..enums import GuardrailExecutionStage
from ..models import GuardrailAction
from ._base import (
    GuardrailMetadata,
    _apply_llm_input_guardrail,
    _store_guardrail_metadata,
    _wrap_function_with_guardrail,
)


def _wrap_llm_with_prompt_injection_guardrail(
    llm: BaseChatModel, metadata: GuardrailMetadata
) -> BaseChatModel:
    """Wrap an LLM to apply prompt injection detection at PRE stage only.

    Prompt injection is evaluated on the input messages before the LLM is called.
    Only GuardrailExecutionStage.PRE is valid; POST and PRE_AND_POST are rejected
    at configuration time by _apply_prompt_injection_guardrail.

    Args:
        llm: BaseChatModel instance to wrap.
        metadata: GuardrailMetadata with the BuiltInValidatorGuardrail and action.

    Returns:
        The same LLM with wrapped invoke/ainvoke.
    """
    _guardrail_opt = metadata.guardrail
    if _guardrail_opt is None:
        return llm
    guardrail: BuiltInValidatorGuardrail = _guardrail_opt
    action = metadata.config["action"]
    guardrail_name = metadata.name
    uipath = UiPath()

    ConcreteType = type(llm)

    class _GuardedLLM(ConcreteType):  # type: ignore[valid-type, misc]
        def invoke(self, messages, config=None, **kwargs):
            if isinstance(messages, list):
                _apply_llm_input_guardrail(
                    messages, guardrail, uipath, action, guardrail_name
                )
            return super().invoke(messages, config, **kwargs)

        async def ainvoke(self, messages, config=None, **kwargs):
            if isinstance(messages, list):
                _apply_llm_input_guardrail(
                    messages, guardrail, uipath, action, guardrail_name
                )
            return await super().ainvoke(messages, config, **kwargs)

    llm.__class__ = _GuardedLLM
    return llm


def _create_prompt_injection_guardrail(
    threshold: float,
    action: GuardrailAction,
    name: str,
    description: str | None,
    enabled_for_evals: bool,
) -> BuiltInValidatorGuardrail:
    """Create a BuiltInValidatorGuardrail for prompt injection detection."""
    return BuiltInValidatorGuardrail(
        id=str(uuid4()),
        name=name,
        description=description
        or f"Detects prompt injection with threshold {threshold}",
        enabled_for_evals=enabled_for_evals,
        selector=GuardrailSelector(scopes=[GuardrailScope.LLM]),
        guardrail_type="builtInValidator",
        validator_type="prompt_injection",
        validator_parameters=[
            NumberParameterValue(
                parameter_type="number",
                id="threshold",
                value=threshold,
            ),
        ],
    )


def _apply_prompt_injection_guardrail(
    obj: Callable[..., Any] | BaseChatModel,
    threshold: float,
    action: GuardrailAction,
    name: str,
    description: str | None,
    enabled_for_evals: bool,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE,
) -> Callable[..., Any] | BaseChatModel:
    """Apply prompt injection guardrail to an object."""
    if action is None:
        raise ValueError("action must be provided")
    if not isinstance(action, GuardrailAction):
        raise ValueError("action must be an instance of GuardrailAction")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
    if not isinstance(enabled_for_evals, bool):
        raise ValueError("enabled_for_evals must be a boolean")

    # Validate stage — prompt injection is an input-only concern
    if stage != GuardrailExecutionStage.PRE:
        from pydantic import BaseModel, field_validator

        class _StageValidator(BaseModel):
            stage: GuardrailExecutionStage

            @field_validator("stage")
            @classmethod
            def _check(cls, v: GuardrailExecutionStage) -> GuardrailExecutionStage:
                if v != GuardrailExecutionStage.PRE:
                    raise ValueError(
                        "prompt_injection_guardrail only supports "
                        "GuardrailExecutionStage.PRE; prompt injection is an "
                        "input-only concern and cannot be evaluated POST-execution."
                    )
                return v

        _StageValidator(stage=stage)

    # Prompt injection only supported at LLM scope
    scope = GuardrailScope.LLM

    guardrail = _create_prompt_injection_guardrail(
        threshold, action, name, description, enabled_for_evals
    )

    metadata = GuardrailMetadata(
        guardrail_type="prompt_injection",
        scope=scope,
        config={
            "threshold": threshold,
            "action": action,
            "stage": stage,
            "enabled_for_evals": enabled_for_evals,
        },
        name=name,
        description=description,
        guardrail=guardrail,
        wrap_llm=_wrap_llm_with_prompt_injection_guardrail,
    )

    _store_guardrail_metadata(obj, metadata)

    if isinstance(obj, BaseChatModel):
        return _wrap_llm_with_prompt_injection_guardrail(obj, metadata)
    if callable(obj):
        return _wrap_function_with_guardrail(obj, metadata)
    return obj


def prompt_injection_guardrail(
    func: Callable[..., Any] | BaseChatModel | None = None,
    *,
    threshold: float = 0.5,
    action: GuardrailAction | None = None,
    name: str = "Prompt Injection Detection",
    description: str | None = None,
    enabled_for_evals: bool = True,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE,
):
    """Decorator for prompt injection detection guardrails.

    Can be applied to LLM instances or factory functions that return LLM instances.
    Prompt injection guardrails are LLM-only.

    Args:
        func: Object to decorate (when used without parentheses).
        threshold: Detection confidence threshold (0.0 to 1.0), default 0.5.
        action: Action to take when prompt injection is detected (required).
        name: Optional name for the guardrail.
        description: Optional description for the guardrail.
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
            Defaults to True.
        stage: When to evaluate the guardrail. Only GuardrailExecutionStage.PRE is
            supported — prompt injection is an input-only concern. Passing POST or
            PRE_AND_POST raises a pydantic.ValidationError. Defaults to PRE.

    Example::

        @prompt_injection_guardrail(
            threshold=0.5,
            action=BlockAction(),
            name="LLM Prompt Injection Detection",
        )
        def create_llm():
            return UiPathChat(model="gpt-4o")

    Returns:
        Decorated function or object
    """
    if func is None:

        def decorator(
            f: Callable[..., Any] | BaseChatModel,
        ) -> Callable[..., Any] | BaseChatModel:
            return _apply_prompt_injection_guardrail(
                f,
                threshold,
                cast(GuardrailAction, action),
                name,
                description,
                enabled_for_evals,
                stage,
            )

        return decorator
    else:
        if action is None:
            raise ValueError(
                "When using @prompt_injection_guardrail without parentheses, "
                "you must provide action as a keyword argument."
            )
        return _apply_prompt_injection_guardrail(
            func, threshold, action, name, description, enabled_for_evals, stage
        )
