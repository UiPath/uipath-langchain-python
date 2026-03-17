"""PII detection guardrail decorator."""

import logging
from typing import Any, Callable, Sequence, cast
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from uipath.core.guardrails import GuardrailScope, GuardrailSelector
from uipath.platform import UiPath
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    EnumListParameterValue,
    MapEnumParameterValue,
)

from ..enums import GuardrailExecutionStage
from ..middlewares._utils import create_modified_tool_result
from ..models import GuardrailAction, PIIDetectionEntity
from ._base import (
    GuardrailMetadata,
    _apply_llm_input_guardrail,
    _apply_llm_output_guardrail,
    _detect_scope,
    _evaluate_guardrail,
    _extract_input,
    _extract_output,
    _handle_guardrail_result,
    _rewrap_input,
    _store_guardrail_metadata,
    _wrap_function_with_guardrail,
)

logger = logging.getLogger(__name__)


def _wrap_tool_with_pii_guardrail(
    tool: BaseTool, metadata: GuardrailMetadata
) -> BaseTool:
    """Wrap a BaseTool to apply PII detection on input (PRE), output (POST), or both.

    The stage is controlled by metadata.config["stage"] (default: PRE_AND_POST).

    Args:
        tool: BaseTool instance to wrap.
        metadata: GuardrailMetadata with the BuiltInValidatorGuardrail and action.

    Returns:
        The same tool with wrapped invoke/ainvoke.
    """
    action = metadata.config["action"]
    guardrail_name = metadata.name
    api_guardrail = metadata.guardrail

    _stage = metadata.config.get("stage", GuardrailExecutionStage.PRE_AND_POST)
    uipath: UiPath | None = None

    def _get_uipath() -> UiPath:
        nonlocal uipath
        if uipath is None:
            uipath = UiPath()
        return uipath

    def _apply_pre(tool_input: Any) -> Any:
        if api_guardrail is None:
            return tool_input
        input_data = _extract_input(tool_input)
        try:
            result = _evaluate_guardrail(input_data, api_guardrail, _get_uipath())
            modified = _handle_guardrail_result(
                result, input_data, action, guardrail_name
            )
            if modified is not None and isinstance(modified, dict):
                return _rewrap_input(tool_input, modified)
        except Exception as e:
            logger.error(
                f"Error evaluating PII guardrail (pre) for tool '{tool.name}': {e}",
                exc_info=True,
            )
        return tool_input

    def _apply_post(tool_input: Any, result: Any) -> Any:
        if api_guardrail is None:
            return result
        output_data = _extract_output(result)
        try:
            eval_result = _evaluate_guardrail(output_data, api_guardrail, _get_uipath())
            modified = _handle_guardrail_result(
                eval_result, output_data, action, guardrail_name
            )
            if modified is not None:
                if isinstance(result, (ToolMessage, Command)):
                    return create_modified_tool_result(result, modified)
                return modified
        except Exception as e:
            logger.error(
                f"Error evaluating PII guardrail (post) for tool '{tool.name}': {e}",
                exc_info=True,
            )
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

    tool.__class__ = _GuardedTool
    return tool


def _wrap_llm_with_pii_guardrail(
    llm: BaseChatModel, metadata: GuardrailMetadata
) -> BaseChatModel:
    """Wrap an LLM to apply PII detection on input (PRE), output (POST), or both.

    The stage is controlled by metadata.config["stage"] (default: PRE_AND_POST).

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
    _stage = metadata.config.get("stage", GuardrailExecutionStage.PRE_AND_POST)
    uipath = UiPath()

    ConcreteType = type(llm)

    class _GuardedLLM(ConcreteType):  # type: ignore[valid-type, misc]
        def invoke(self, messages, config=None, **kwargs):
            if isinstance(messages, list) and _stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_input_guardrail(
                    messages, guardrail, uipath, action, guardrail_name
                )
            response = super().invoke(messages, config, **kwargs)
            if isinstance(response, AIMessage) and _stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_output_guardrail(
                    response, guardrail, uipath, action, guardrail_name
                )
            return response

        async def ainvoke(self, messages, config=None, **kwargs):
            if isinstance(messages, list) and _stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_input_guardrail(
                    messages, guardrail, uipath, action, guardrail_name
                )
            response = await super().ainvoke(messages, config, **kwargs)
            if isinstance(response, AIMessage) and _stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_output_guardrail(
                    response, guardrail, uipath, action, guardrail_name
                )
            return response

    llm.__class__ = _GuardedLLM
    return llm


def _create_pii_guardrail(
    entities: Sequence[PIIDetectionEntity],
    action: GuardrailAction,
    name: str,
    description: str | None,
    scope: GuardrailScope,
    enabled_for_evals: bool,
) -> BuiltInValidatorGuardrail:
    """Create a BuiltInValidatorGuardrail for PII detection."""
    entity_names = [entity.name for entity in entities]
    entity_thresholds = {entity.name: entity.threshold for entity in entities}

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

    selector_kwargs: dict[str, Any] = {"scopes": [scope]}

    return BuiltInValidatorGuardrail(
        id=str(uuid4()),
        name=name,
        description=description or f"Detects PII entities: {', '.join(entity_names)}",
        enabled_for_evals=enabled_for_evals,
        selector=GuardrailSelector(**selector_kwargs),
        guardrail_type="builtInValidator",
        validator_type="pii_detection",
        validator_parameters=validator_parameters,
    )


def _apply_pii_guardrail(
    obj: Callable[..., Any] | BaseChatModel | BaseTool,
    entities: Sequence[PIIDetectionEntity],
    action: GuardrailAction,
    name: str,
    description: str | None,
    enabled_for_evals: bool,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
) -> Callable[..., Any] | BaseChatModel | BaseTool:
    """Apply PII guardrail to an object (LLM, tool, or callable)."""
    if not entities:
        raise ValueError("entities must be provided and non-empty")
    if action is None:
        raise ValueError("action must be provided")
    if not isinstance(action, GuardrailAction):
        raise ValueError("action must be an instance of GuardrailAction")

    scope = _detect_scope(obj)
    if not isinstance(enabled_for_evals, bool):
        raise ValueError("enabled_for_evals must be a boolean")

    guardrail = _create_pii_guardrail(
        entities, action, name, description, scope, enabled_for_evals
    )

    metadata = GuardrailMetadata(
        guardrail_type="pii",
        scope=scope,
        config={
            "entities": list(entities),
            "action": action,
            "stage": stage,
            "enabled_for_evals": enabled_for_evals,
        },
        name=name,
        description=description,
        guardrail=guardrail,
        wrap_tool=_wrap_tool_with_pii_guardrail,
        wrap_llm=_wrap_llm_with_pii_guardrail,
    )

    _store_guardrail_metadata(obj, metadata)

    if isinstance(obj, BaseTool):
        return _wrap_tool_with_pii_guardrail(obj, metadata)
    if isinstance(obj, BaseChatModel):
        return _wrap_llm_with_pii_guardrail(obj, metadata)
    if callable(obj):
        return _wrap_function_with_guardrail(obj, metadata)
    return obj


def pii_detection_guardrail(
    func: Callable[..., Any] | BaseChatModel | BaseTool | None = None,
    *,
    entities: Sequence[PIIDetectionEntity] | None = None,
    action: GuardrailAction | None = None,
    name: str = "PII Detection",
    description: str | None = None,
    enabled_for_evals: bool = True,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
):
    """Decorator for PII detection guardrails.

    Can be applied to LLM instances, agent factory functions, or individual tools.
    Scope is automatically detected from the decorated object.

    Args:
        func: Object to decorate (when used without parentheses).
        entities: List of PII entities to detect (required).
        action: Action to take when PII is detected (required).
        name: Optional name for the guardrail.
        description: Optional description for the guardrail.
        enabled_for_evals: Whether this guardrail is enabled for evaluation scenarios.
            Defaults to True.
        stage: When to evaluate the guardrail relative to execution. One of
            GuardrailExecutionStage.PRE (input only), POST (output only), or
            PRE_AND_POST (both). Defaults to PRE_AND_POST.

    Example::

        # Apply to LLM factory function
        @pii_detection_guardrail(
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            stage=GuardrailExecutionStage.PRE,
        )
        def create_llm():
            return UiPathChat(model="gpt-4o")

        # Apply to a specific tool
        analyze_joke_syntax = pii_detection_guardrail(
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            name="Tool PII Detection",
        )(analyze_joke_syntax)

    Returns:
        Decorated function or object
    """

    def _apply(
        obj: Callable[..., Any] | BaseChatModel | BaseTool,
    ) -> Callable[..., Any] | BaseChatModel | BaseTool:
        result = _apply_pii_guardrail(
            obj,
            cast(Sequence[PIIDetectionEntity], entities),
            cast(GuardrailAction, action),
            name,
            description,
            enabled_for_evals,
            stage,
        )
        return result

    if func is None:
        # Called as @pii_detection_guardrail(...) — return decorator
        return _apply
    else:
        # Called as @pii_detection_guardrail (bare, no parentheses)
        if entities is None or action is None:
            raise ValueError(
                "When using @pii_detection_guardrail without parentheses, "
                "you must provide entities and action as keyword arguments."
            )
        return _apply(func)
