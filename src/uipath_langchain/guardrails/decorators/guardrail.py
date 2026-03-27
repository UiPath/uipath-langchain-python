"""Single @guardrail decorator for all guardrail types."""

import inspect
import logging
from functools import wraps
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailValidationResult,
)
from uipath.platform import UiPath
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from ...agent.exceptions import AgentRuntimeError
from ..enums import GuardrailExecutionStage
from ..middlewares._utils import create_modified_tool_result
from ..models import GuardrailAction
from ._base import (
    _apply_message_text_modification,
    _detect_scope,
    _evaluate_guardrail,
    _extract_input,
    _extract_message_text,
    _extract_output,
    _get_last_ai_message,
    _get_last_human_message,
    _handle_guardrail_result,
    _rewrap_input,
)
from .validators._base import GuardrailValidatorBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluator factory
# ---------------------------------------------------------------------------

_EvaluatorFn = Callable[
    [
        "str | dict[str, Any]",  # data
        GuardrailExecutionStage,  # stage
        "dict[str, Any] | None",  # input_data
        "dict[str, Any] | None",  # output_data
    ],
    GuardrailValidationResult,
]


def _make_evaluator(
    validator: GuardrailValidatorBase,
    built_in_guardrail: BuiltInValidatorGuardrail | None,
) -> _EvaluatorFn:
    """Return a unified evaluation callable for use in all wrappers.

    If *built_in_guardrail* is provided the callable hits the UiPath API (lazily
    initialising ``UiPath()``). Otherwise it delegates to ``validator.evaluate()``.

    Args:
        validator: The validator instance (used for local evaluation path).
        built_in_guardrail: Pre-built ``BuiltInValidatorGuardrail``, or ``None``.

    Returns:
        Callable with signature ``(data, stage, input_data, output_data)``.
    """
    if built_in_guardrail is not None:
        _uipath_holder: list[UiPath] = []

        def _api_eval(
            data: str | dict[str, Any],
            stage: GuardrailExecutionStage,
            input_data: dict[str, Any] | None,
            output_data: dict[str, Any] | None,
        ) -> GuardrailValidationResult:
            if not _uipath_holder:
                _uipath_holder.append(UiPath())
            return _evaluate_guardrail(data, built_in_guardrail, _uipath_holder[0])

        return _api_eval

    def _local_eval(
        data: str | dict[str, Any],
        stage: GuardrailExecutionStage,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
    ) -> GuardrailValidationResult:
        return validator.evaluate(data, stage, input_data, output_data)

    return _local_eval


# ---------------------------------------------------------------------------
# Generic tool wrapper
# ---------------------------------------------------------------------------


def _wrap_tool_with_guardrail(
    tool: BaseTool,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> BaseTool:
    """Wrap a ``BaseTool`` to apply the guardrail at PRE, POST, or PRE_AND_POST.

    Uses Pydantic ``__class__`` swapping so all Pydantic fields and the
    ``StructuredTool`` interface are fully inherited by the guarded subclass.

    Args:
        tool: ``BaseTool`` instance to wrap.
        evaluator: Unified evaluation callable from ``_make_evaluator()``.
        action: Action to invoke on validation failure.
        name: Guardrail name (used in action and log messages).
        stage: When to run the guardrail (PRE, POST, or PRE_AND_POST).

    Returns:
        The same tool object with its ``__class__`` swapped to a guarded subclass.
    """
    _stage = stage

    def _apply_pre(tool_input: Any) -> Any:
        input_data = _extract_input(tool_input)
        try:
            result = evaluator(
                input_data, GuardrailExecutionStage.PRE, input_data, None
            )
        except Exception as exc:
            logger.error(
                "Error evaluating guardrail (pre) for tool %r: %s",
                tool.name,
                exc,
                exc_info=True,
            )
            return tool_input
        try:
            modified = _handle_guardrail_result(result, input_data, action, name)
        except AgentRuntimeError:
            raise
        if modified is not None and isinstance(modified, dict):
            return _rewrap_input(tool_input, modified)
        return tool_input

    def _apply_post(tool_input: Any, raw_result: Any) -> Any:
        input_data = _extract_input(tool_input)
        output_data = _extract_output(raw_result)
        try:
            result = evaluator(
                output_data, GuardrailExecutionStage.POST, input_data, output_data
            )
        except Exception as exc:
            logger.error(
                "Error evaluating guardrail (post) for tool %r: %s",
                tool.name,
                exc,
                exc_info=True,
            )
            return raw_result
        try:
            modified = _handle_guardrail_result(result, output_data, action, name)
        except AgentRuntimeError:
            raise
        if modified is not None:
            if isinstance(raw_result, (ToolMessage, Command)):
                return create_modified_tool_result(raw_result, modified)
            return modified
        return raw_result

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

        # ainvoke is intentionally NOT overridden here.
        # StructuredTool.ainvoke (for sync tools without a coroutine) delegates to
        # self.invoke via run_in_executor. Overriding ainvoke would cause the POST
        # guardrail to fire twice: once inside self.invoke and once after super().ainvoke()
        # returns. Guardrails are applied correctly through invoke alone.

    tool.__class__ = _GuardedTool
    return tool


# ---------------------------------------------------------------------------
# Generic LLM wrapper
# ---------------------------------------------------------------------------


def _wrap_llm_with_guardrail(
    llm: BaseChatModel,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> BaseChatModel:
    """Wrap a ``BaseChatModel`` to apply the guardrail at PRE, POST, or PRE_AND_POST.

    PRE: evaluates the last ``HumanMessage`` before the LLM is called.
    POST: evaluates the ``AIMessage`` response after the LLM returns.

    Args:
        llm: ``BaseChatModel`` instance to wrap.
        evaluator: Unified evaluation callable from ``_make_evaluator()``.
        action: Action to invoke on validation failure.
        name: Guardrail name.
        stage: When to run the guardrail.

    Returns:
        The same LLM object with its ``__class__`` swapped to a guarded subclass.
    """
    _stage = stage

    ConcreteType = type(llm)

    class _GuardedLLM(ConcreteType):  # type: ignore[valid-type, misc]
        def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> Any:
            if isinstance(messages, list) and _stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_pre(messages, evaluator, action, name)
            response = super().invoke(messages, config, **kwargs)
            if isinstance(response, AIMessage) and _stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_post(response, evaluator, action, name)
            return response

        async def ainvoke(
            self, messages: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            if isinstance(messages, list) and _stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_pre(messages, evaluator, action, name)
            response = await super().ainvoke(messages, config, **kwargs)
            if isinstance(response, AIMessage) and _stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_llm_post(response, evaluator, action, name)
            return response

    llm.__class__ = _GuardedLLM
    return llm


def _apply_llm_pre(
    messages: list[BaseMessage],
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
) -> None:
    """Evaluate the last HumanMessage in-place (PRE stage, LLM scope)."""
    msg = _get_last_human_message(messages)
    if msg is None:
        return
    text = _extract_message_text(msg)
    if not text:
        return
    try:
        result = evaluator(text, GuardrailExecutionStage.PRE, None, None)
    except Exception:
        return
    modified = _handle_guardrail_result(result, text, action, name)
    if isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


def _apply_llm_post(
    response: AIMessage,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
) -> None:
    """Evaluate the AIMessage content in-place (POST stage, LLM scope)."""
    if not isinstance(response.content, str) or not response.content:
        return
    try:
        result = evaluator(response.content, GuardrailExecutionStage.POST, None, None)
    except Exception:
        return
    modified = _handle_guardrail_result(result, response.content, action, name)
    if isinstance(modified, str) and modified != response.content:
        response.content = modified


# ---------------------------------------------------------------------------
# Generic StateGraph / CompiledStateGraph wrappers (AGENT scope)
# ---------------------------------------------------------------------------


def _wrap_stategraph_with_guardrail(
    graph: StateGraph[Any, Any],
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> StateGraph[Any, Any]:
    """Wrap a ``StateGraph``'s invoke/ainvoke to apply the guardrail (AGENT scope).

    Args:
        graph: ``StateGraph`` instance to wrap.
        evaluator: Unified evaluation callable.
        action: Action to invoke on validation failure.
        name: Guardrail name.
        stage: When to run the guardrail.

    Returns:
        The same graph with patched ``invoke`` / ``ainvoke`` methods.
    """
    if hasattr(graph, "invoke"):
        original_invoke = graph.invoke

        @wraps(original_invoke)
        def wrapped_invoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
            if stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_agent_input_guardrail(input, evaluator, action, name)
            output = original_invoke(input, config, **kwargs)
            if stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_agent_output_guardrail(output, evaluator, action, name)
            return output

        graph.invoke = wrapped_invoke

    if hasattr(graph, "ainvoke"):
        original_ainvoke = graph.ainvoke

        @wraps(original_ainvoke)
        async def wrapped_ainvoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
            if stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_agent_input_guardrail(input, evaluator, action, name)
            output = await original_ainvoke(input, config, **kwargs)
            if stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                _apply_agent_output_guardrail(output, evaluator, action, name)
            return output

        graph.ainvoke = wrapped_ainvoke

    return graph


def _wrap_compiled_graph_with_guardrail(
    graph: CompiledStateGraph[Any, Any, Any],
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> CompiledStateGraph[Any, Any, Any]:
    """Wrap a ``CompiledStateGraph``'s invoke/ainvoke (AGENT scope).

    Args:
        graph: ``CompiledStateGraph`` instance to wrap.
        evaluator: Unified evaluation callable.
        action: Action to invoke on validation failure.
        name: Guardrail name.
        stage: When to run the guardrail.

    Returns:
        The same compiled graph with patched ``invoke`` / ``ainvoke`` methods.
    """
    original_invoke = graph.invoke
    original_ainvoke = graph.ainvoke

    @wraps(original_invoke)
    def wrapped_invoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        if stage in (
            GuardrailExecutionStage.PRE,
            GuardrailExecutionStage.PRE_AND_POST,
        ):
            _apply_agent_input_guardrail(input, evaluator, action, name)
        output = original_invoke(input, config, **kwargs)
        if stage in (
            GuardrailExecutionStage.POST,
            GuardrailExecutionStage.PRE_AND_POST,
        ):
            _apply_agent_output_guardrail(output, evaluator, action, name)
        return output

    @wraps(original_ainvoke)
    async def wrapped_ainvoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        if stage in (
            GuardrailExecutionStage.PRE,
            GuardrailExecutionStage.PRE_AND_POST,
        ):
            _apply_agent_input_guardrail(input, evaluator, action, name)
        output = await original_ainvoke(input, config, **kwargs)
        if stage in (
            GuardrailExecutionStage.POST,
            GuardrailExecutionStage.PRE_AND_POST,
        ):
            _apply_agent_output_guardrail(output, evaluator, action, name)
        return output

    graph.invoke = wrapped_invoke  # type: ignore[method-assign]
    graph.ainvoke = wrapped_ainvoke  # type: ignore[method-assign]
    return graph


def _apply_agent_input_guardrail(
    input: Any,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
) -> None:
    """Evaluate the last HumanMessage from agent input in-place."""
    if not isinstance(input, dict) or "messages" not in input:
        return
    messages = input["messages"]
    if not isinstance(messages, list):
        return
    msg = _get_last_human_message(messages)
    if msg is None:
        return
    text = _extract_message_text(msg)
    if not text:
        return
    try:
        result = evaluator(text, GuardrailExecutionStage.PRE, None, None)
    except Exception:
        return
    modified = _handle_guardrail_result(result, text, action, name)
    if isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


def _apply_agent_output_guardrail(
    output: Any,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
) -> None:
    """Evaluate the last AIMessage from agent output in-place."""
    if not isinstance(output, dict) or "messages" not in output:
        return
    messages = output["messages"]
    if not isinstance(messages, list):
        return
    msg = _get_last_ai_message(messages)
    if msg is None:
        return
    text = _extract_message_text(msg)
    if not text:
        return
    try:
        result = evaluator(text, GuardrailExecutionStage.POST, None, None)
    except Exception:
        return
    modified = _handle_guardrail_result(result, text, action, name)
    if isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


# ---------------------------------------------------------------------------
# Factory function wrapper (AGENT scope: function returning LLM/tool/graph)
# ---------------------------------------------------------------------------


def _wrap_factory_function(
    func: Callable[..., Any],
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> Callable[..., Any]:
    """Wrap a factory function, applying the guardrail to its return value.

    After calling the function the return type is inspected and delegated to
    the appropriate typed wrapper.

    Args:
        func: Factory function to wrap.
        evaluator: Unified evaluation callable.
        action: Action to invoke on validation failure.
        name: Guardrail name.
        stage: When to run the guardrail.

    Returns:
        Wrapped function (sync or async, matching the original).
    """

    def _dispatch(result: Any) -> Any:
        if isinstance(result, CompiledStateGraph):
            return _wrap_compiled_graph_with_guardrail(
                result, evaluator, action, name, stage
            )
        if isinstance(result, StateGraph):
            return _wrap_stategraph_with_guardrail(
                result, evaluator, action, name, stage
            )
        if isinstance(result, BaseChatModel):
            return _wrap_llm_with_guardrail(result, evaluator, action, name, stage)
        if isinstance(result, BaseTool):
            return _wrap_tool_with_guardrail(result, evaluator, action, name, stage)
        return result

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapped_async(*args: Any, **kwargs: Any) -> Any:
            return _dispatch(await func(*args, **kwargs))

        return wrapped_async

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _dispatch(func(*args, **kwargs))

    return wrapped


# ---------------------------------------------------------------------------
# Public @guardrail decorator
# ---------------------------------------------------------------------------


def guardrail(
    func: Any = None,
    *,
    validator: GuardrailValidatorBase,
    action: GuardrailAction,
    name: str = "Guardrail",
    description: str | None = None,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
    enabled_for_evals: bool = True,
) -> Any:
    """Apply a guardrail to an LLM, tool, or agent factory function.

    The guardrail is described by a *validator* (what to check) and an *action*
    (how to respond on violation). Scope is auto-detected from the decorated
    object type, and validated against the validator's ``supported_scopes`` /
    ``supported_stages`` at decoration time.

    Can be stacked: multiple ``@guardrail`` decorators on the same object chain
    via Pydantic ``__class__`` swapping (tools/LLMs) or function wrapping (agents).

    Args:
        func: Object to decorate when used without parentheses (rare).
        validator: ``GuardrailValidatorBase`` instance defining what to check.
        action: ``GuardrailAction`` instance defining how to respond on violation.
        name: Human-readable name for this guardrail instance.
        description: Optional description (used when building API guardrails).
        stage: When to evaluate — ``PRE``, ``POST``, or ``PRE_AND_POST``.
            Defaults to ``PRE_AND_POST``.
        enabled_for_evals: Whether this guardrail is active in evaluation
            scenarios. Defaults to ``True``.

    Returns:
        The decorated object (same type as input).

    Raises:
        ValueError: If the validator does not support the detected scope or
            the requested stage, or if ``action`` is missing / invalid.

    Example::

        pii = PIIValidator(entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)])

        @guardrail(validator=pii, action=LogAction(), name="LLM PII", stage=GuardrailExecutionStage.PRE)
        def create_llm():
            return UiPathChat(model="gpt-4o")

        @guardrail(validator=PIIValidator(entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)]),
                   action=BlockAction(), name="Tool PII")
        @tool
        def my_tool(text: str) -> str: ...
    """
    if action is None:
        raise ValueError("action must be provided")
    if not isinstance(action, GuardrailAction):
        raise ValueError("action must be an instance of GuardrailAction")
    if not isinstance(enabled_for_evals, bool):
        raise ValueError("enabled_for_evals must be a boolean")

    def _apply(obj: Any) -> Any:
        # Factory functions (plain callables that are not tool/LLM/graph instances)
        # have an unknown scope until they are called and their return type is known.
        # Handle them separately to avoid false scope-validation failures.
        _is_factory = callable(obj) and not isinstance(
            obj, (BaseTool, BaseChatModel, StateGraph, CompiledStateGraph)
        )

        if _is_factory:
            # TOOL-only validators must be applied directly to @tool instances,
            # not to factory functions — the scope is unresolvable at decoration time.
            if validator.supported_scopes and all(
                s.value == "Tool" for s in validator.supported_scopes
            ):
                raise ValueError(
                    f"@guardrail with {type(validator).__name__} can only be applied "
                    "to BaseTool instances (decorated with @tool). "
                    "Apply it directly to the tool, not to a factory function."
                )
            validator.validate_stage(stage)
            # Use the validator's primary scope (first supported, or AGENT if
            # unrestricted) to build the API guardrail instance.
            api_scope = (
                validator.supported_scopes[0]
                if validator.supported_scopes
                else _detect_scope(obj)
            )
        else:
            scope = _detect_scope(obj)
            validator.validate_scope(scope)
            validator.validate_stage(stage)
            api_scope = scope

        built_in_guardrail = validator.build_built_in_guardrail(
            api_scope, name, description, enabled_for_evals
        )
        evaluator = _make_evaluator(validator, built_in_guardrail)

        if isinstance(obj, BaseTool):
            return _wrap_tool_with_guardrail(obj, evaluator, action, name, stage)
        if isinstance(obj, BaseChatModel):
            return _wrap_llm_with_guardrail(obj, evaluator, action, name, stage)
        if isinstance(obj, CompiledStateGraph):
            return _wrap_compiled_graph_with_guardrail(
                obj, evaluator, action, name, stage
            )
        if isinstance(obj, StateGraph):
            return _wrap_stategraph_with_guardrail(obj, evaluator, action, name, stage)
        if callable(obj):
            return _wrap_factory_function(obj, evaluator, action, name, stage)
        raise ValueError(
            f"@guardrail cannot be applied to {type(obj)!r}. "
            "Target must be a BaseTool, BaseChatModel, StateGraph, "
            "CompiledStateGraph, or a callable factory function."
        )

    if func is None:
        return _apply
    return _apply(func)
