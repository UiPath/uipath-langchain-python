"""LangChain/LangGraph adapter for the ``@guardrail`` decorator.

Registers a :class:`LangChainGuardrailAdapter` that teaches the platform
``@guardrail`` decorator how to wrap LangChain and LangGraph objects:

- ``BaseTool`` → TOOL scope
- ``BaseChatModel`` → LLM scope
- ``StateGraph`` / ``CompiledStateGraph`` → AGENT scope

The adapter is auto-registered when ``uipath_langchain.guardrails`` is imported.
"""

import asyncio
import json
import logging
from collections.abc import Iterable
from contextvars import ContextVar
from functools import wraps
from itertools import chain
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import GuardrailAttachment
from uipath.platform.guardrails.attachments import (
    MAX_ATTACHMENTS,
    discover_attachments,
)
from uipath.platform.guardrails.decorators._core import (
    _EvaluatorFn,
    _extract_input,
    _extract_output,
    _rewrap_input,
)
from uipath.platform.guardrails.decorators._enums import GuardrailExecutionStage
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException
from uipath.platform.guardrails.decorators._models import GuardrailAction
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.guardrails._action_context import (
    GuardrailActionContext,
    _action_context,
    component_label,
)
from uipath_langchain.guardrails.middlewares._attachments import (
    JOB_ATTACHMENTS_STATE_KEY,
    NO_TEXT_PLACEHOLDER,
    build_registry,
    output_references,
    registry_references,
)
from uipath_langchain.guardrails.middlewares._utils import create_modified_tool_result

logger = logging.getLogger(__name__)

#: Logged when a guardrail evaluator raises. Evaluation is fail-open, so the
#: run continues unvalidated -- the WARNING (with traceback) is the only signal
#: that a guardrail is misconfigured. Takes the guardrail name as its argument.
_EVALUATION_FAILED_MSG = (
    "Guardrail '%s' evaluation failed; continuing without validation."
)

_NOT_AGENT_OUTPUT_KEYS = frozenset(
    {"messages", "structured_response", JOB_ATTACHMENTS_STATE_KEY}
)

#: The tool guards whose ``ainvoke`` is running, so the ``invoke`` it reaches skips.
_AINVOKE_GUARDS: ContextVar[frozenset[object]] = ContextVar(
    "_AINVOKE_GUARDS", default=frozenset()
)


# ---------------------------------------------------------------------------
# Exception conversion helper
# ---------------------------------------------------------------------------


def _convert_block_exception(exc: GuardrailBlockException) -> AgentRuntimeError:
    """Convert a :class:`GuardrailBlockException` to :class:`AgentRuntimeError`."""
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION,
        title=exc.title,
        detail=exc.detail,
        category=UiPathErrorCategory.USER,
    )


# ---------------------------------------------------------------------------
# Guardrail action invocation (with runtime context publishing)
# ---------------------------------------------------------------------------


def _run_action(
    action: GuardrailAction,
    result: Any,
    data: Any,
    name: str,
    *,
    scope: GuardrailScope | None,
    stage: GuardrailExecutionStage | None,
    component: str | None,
    input_payload: str | None = None,
) -> str | dict[str, Any] | None:
    """Run a guardrail action with its runtime context published.

    Publishes the guardrail context (scope / stage / component / input_payload)
    on a ``ContextVar`` for the duration of the action call, so context-aware
    actions (e.g. ``EscalateAction``) can read it at runtime instead of requiring
    it to be hardcoded. ``input_payload`` carries the original input at POST so
    the action can show both input and output. Mirrors the middleware path's
    publishing in ``_base.py``.

    A :class:`GuardrailBlockException` is converted to an ``AgentRuntimeError``.
    Any other exception — notably LangGraph's ``GraphInterrupt`` raised by an
    escalation action — propagates unchanged so the run can suspend.
    """
    token = _action_context.set(
        GuardrailActionContext(
            scope=scope,
            execution_stage=stage,
            component=component,
            input_payload=input_payload,
        )
    )
    try:
        return action.handle_validation_result(result, data, name)
    except GuardrailBlockException as exc:
        raise _convert_block_exception(exc) from exc
    finally:
        _action_context.reset(token)


# ---------------------------------------------------------------------------
# Evaluation with attachments
# ---------------------------------------------------------------------------


def _evaluate(
    evaluator: _EvaluatorFn,
    data: str | dict[str, Any],
    stage: GuardrailExecutionStage,
    input_data: dict[str, Any] | None,
    output_data: dict[str, Any] | None,
    attachments: list[GuardrailAttachment] | None,
) -> GuardrailValidationResult:
    """Call the evaluator, passing ``attachments`` when there is something to say.

    A non-empty list is always passed. An empty list (files apply, none found) is
    passed only to an evaluator that declares ``forwards_attachments``, which
    ``@guardrail`` sets; ``None`` (say nothing about files) is never passed. An
    evaluator written before files were supported takes four positional
    arguments, so a call without files keeps that signature.
    """
    if attachments or (
        attachments is not None
        and getattr(evaluator, "forwards_attachments", None) is True
    ):
        return evaluator(data, stage, input_data, output_data, attachments=attachments)
    return evaluator(data, stage, input_data, output_data)


def _forwards_attachments(evaluator: _EvaluatorFn) -> bool:
    """Tell whether the evaluator sends files on; one that does not say is assumed to.

    ``@guardrail`` marks its evaluator ``forwards_attachments = False`` when every
    file would be dropped (text scope, or a validator that takes no files), so a
    call with files and no text has nothing to evaluate. The uipath-platform
    release that provides ``uipath.platform.guardrails.attachments`` sets it, so
    the default only applies to hand-written evaluators.
    """
    return getattr(evaluator, "forwards_attachments", True) is not False


def _union(
    *groups: Iterable[GuardrailAttachment] | None,
) -> list[GuardrailAttachment] | None:
    """Merge reference lists, keeping the first of each id, up to the API limit.

    ``None`` when any part could not be scanned, so the check says nothing about
    files rather than forwarding a partial set.
    """
    if any(group is None for group in groups):
        return None
    unique: dict[str, GuardrailAttachment] = {}
    for reference in chain.from_iterable(group for group in groups if group):
        unique.setdefault(reference.id, reference)
    return list(unique.values())[:MAX_ATTACHMENTS]


def _agent_input_attachments(
    input: dict[str, Any],
) -> list[GuardrailAttachment] | None:
    """Return the files the run holds plus the attachments typed state keys carry."""
    state_keys = {key: value for key, value in input.items() if key != "messages"}
    return _union(
        registry_references(build_registry(input)), discover_attachments(state_keys)
    )


def _agent_output_attachments(
    output: dict[str, Any], message: AIMessage | None
) -> list[GuardrailAttachment] | None:
    """Return the held files the output names plus the attachments it returns."""
    named = output_references(
        message,
        build_registry(output),
        structured_response=output.get("structured_response"),
    )
    returned = discover_attachments(
        {
            key: value
            for key, value in output.items()
            if key not in _NOT_AGENT_OUTPUT_KEYS
        }
    )
    return _union(named, returned)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _get_last_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    """Return the last HumanMessage in a list, or None if absent."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    return None


def _get_last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    """Return the last AIMessage in a list, or None if absent."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _extract_message_text(msg: BaseMessage) -> str:
    """Extract plain text content from a message."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = [
            part.get("text", "")
            for part in msg.content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(filter(None, parts))
    return ""


def _input_payload_from_messages(
    messages: list[BaseMessage] | None,
) -> str | None:
    """JSON-encode the last HumanMessage's text as the original-input payload.

    Used at POST so the escalation action can show the original input alongside
    the flagged output. Returns ``None`` when there is no usable human input.
    """
    if not messages:
        return None
    msg = _get_last_human_message(messages)
    if msg is None:
        return None
    text = _extract_message_text(msg)
    return json.dumps(text) if text else None


def _apply_message_text_modification(msg: BaseMessage, modified: str) -> None:
    """Apply a modified text string back to a message in-place."""
    if isinstance(msg.content, str):
        msg.content = modified
    elif isinstance(msg.content, list):
        for part in msg.content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = modified
                break


# ---------------------------------------------------------------------------
# LangChain-aware output extraction
# ---------------------------------------------------------------------------


def _lc_extract_output(result: Any) -> dict[str, Any]:
    """Normalise tool output to a dict, handling LangGraph ToolMessage / Command.

    Unwraps ``ToolMessage`` and ``Command`` envelopes first, then delegates to
    the platform's pure-Python ``_extract_output`` for JSON / literal-eval parsing.
    """
    content: Any = result
    if isinstance(result, Command):
        update = result.update if hasattr(result, "update") else {}
        messages = update.get("messages", []) if isinstance(update, dict) else []
        if messages and isinstance(messages[0], ToolMessage):
            content = messages[0].content
        else:
            return {}
    elif isinstance(result, ToolMessage):
        content = result.content

    if content is not result:
        # Delegate to the pure-Python extractor with pre-processed content.
        return _extract_output(content)
    return _extract_output(result)


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


def _wrap_tool_with_guardrail(
    tool: BaseTool,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> BaseTool:
    """Wrap a ``BaseTool`` using Pydantic ``__class__`` swapping.

    Both ``invoke`` and ``ainvoke`` are guarded. ``Tool`` and ``StructuredTool``
    without a coroutine run ``ainvoke`` through ``self.invoke`` in an executor, so
    the ``invoke`` a guarded ``ainvoke`` reaches skips its own check.
    """
    guard = object()
    checks_pre = stage in (
        GuardrailExecutionStage.PRE,
        GuardrailExecutionStage.PRE_AND_POST,
    )
    checks_post = stage in (
        GuardrailExecutionStage.POST,
        GuardrailExecutionStage.PRE_AND_POST,
    )

    def _apply_pre(tool_input: Any) -> Any:
        input_data = _extract_input(tool_input)
        attachments = (
            discover_attachments(input_data) if _forwards_attachments(evaluator) else []
        )
        try:
            result = _evaluate(
                evaluator,
                input_data,
                GuardrailExecutionStage.PRE,
                input_data,
                None,
                attachments,
            )
        except Exception as exc:
            logger.error(
                "Error evaluating guardrail (pre) for tool %r: %s",
                tool.name,
                exc,
                exc_info=True,
            )
            return tool_input
        modified = None
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            modified = _run_action(
                action,
                result,
                input_data,
                name,
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.PRE,
                component=tool.name,
            )
        if modified is not None and isinstance(modified, dict):
            return _rewrap_input(tool_input, modified)
        return tool_input

    def _apply_post(tool_input: Any, raw_result: Any) -> Any:
        input_data = _extract_input(tool_input)
        output_data = _lc_extract_output(raw_result)
        attachments = (
            discover_attachments(output_data)
            if _forwards_attachments(evaluator)
            else []
        )
        try:
            result = _evaluate(
                evaluator,
                output_data,
                GuardrailExecutionStage.POST,
                input_data,
                output_data,
                attachments,
            )
        except Exception as exc:
            logger.error(
                "Error evaluating guardrail (post) for tool %r: %s",
                tool.name,
                exc,
                exc_info=True,
            )
            return raw_result
        modified = None
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            modified = _run_action(
                action,
                result,
                output_data,
                name,
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.POST,
                component=tool.name,
                input_payload=json.dumps(input_data),
            )
        if modified is not None:
            if isinstance(raw_result, (ToolMessage, Command)):
                return create_modified_tool_result(raw_result, modified)
            return modified
        return raw_result

    ConcreteToolType = type(tool)

    class _GuardedTool(ConcreteToolType):  # type: ignore[valid-type, misc]
        def invoke(self, tool_input: Any, config: Any = None, **kwargs: Any) -> Any:
            active = _AINVOKE_GUARDS.get()
            if guard in active:
                token = _AINVOKE_GUARDS.set(active - {guard})
                try:
                    return super().invoke(tool_input, config, **kwargs)
                finally:
                    _AINVOKE_GUARDS.reset(token)
            guarded_input = _apply_pre(tool_input) if checks_pre else tool_input
            result = super().invoke(guarded_input, config, **kwargs)
            if checks_post:
                result = _apply_post(guarded_input, result)
            return result

        async def ainvoke(
            self, tool_input: Any, config: Any = None, **kwargs: Any
        ) -> Any:
            guarded_input = tool_input
            if checks_pre:
                guarded_input = await asyncio.to_thread(_apply_pre, tool_input)
            token = _AINVOKE_GUARDS.set(_AINVOKE_GUARDS.get() | {guard})
            try:
                result = await super().ainvoke(guarded_input, config, **kwargs)
            finally:
                _AINVOKE_GUARDS.reset(token)
            if checks_post:
                result = await asyncio.to_thread(_apply_post, guarded_input, result)
            return result

    tool.__class__ = _GuardedTool
    return tool


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------


def _apply_llm_pre(
    messages: list[BaseMessage],
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
) -> None:
    """Evaluate the last HumanMessage in-place (PRE stage, LLM scope).

    Every file the messages hold is sent with the text; files without text
    evaluate ``NO_TEXT_PLACEHOLDER``, which is never written back. An evaluator
    that drops files gets none, so files alone are not evaluated.
    """
    msg = _get_last_human_message(messages)
    text = _extract_message_text(msg) if msg is not None else ""
    attachments = (
        registry_references(build_registry({"messages": messages}))
        if _forwards_attachments(evaluator)
        else []
    )
    if not text and not attachments:
        return
    payload = text or NO_TEXT_PLACEHOLDER
    try:
        result = _evaluate(
            evaluator, payload, GuardrailExecutionStage.PRE, None, None, attachments
        )
    except Exception:
        logger.warning(_EVALUATION_FAILED_MSG, name, exc_info=True)
        return
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        modified = _run_action(
            action,
            result,
            payload,
            name,
            scope=GuardrailScope.LLM,
            stage=GuardrailExecutionStage.PRE,
            component=component_label(GuardrailScope.LLM),
        )
    if msg is not None and text and isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


def _apply_llm_post(
    response: AIMessage,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    input_messages: list[BaseMessage] | None = None,
) -> None:
    """Evaluate the AIMessage content in-place (POST stage, LLM scope).

    The held files the response names are sent with its text, read from the text
    blocks when the content is a list; files without text evaluate
    ``NO_TEXT_PLACEHOLDER``. Only string content is written back. An evaluator
    that drops files gets none, so files alone are not evaluated.
    """
    text = response.content if isinstance(response.content, str) else ""
    attachments = (
        output_references(response, build_registry({"messages": input_messages or []}))
        if _forwards_attachments(evaluator)
        else []
    )
    if not text and not attachments:
        return
    payload = text or _extract_message_text(response) or NO_TEXT_PLACEHOLDER
    try:
        result = _evaluate(
            evaluator, payload, GuardrailExecutionStage.POST, None, None, attachments
        )
    except Exception:
        logger.warning(_EVALUATION_FAILED_MSG, name, exc_info=True)
        return
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        modified = _run_action(
            action,
            result,
            payload,
            name,
            scope=GuardrailScope.LLM,
            stage=GuardrailExecutionStage.POST,
            component=component_label(GuardrailScope.LLM),
            input_payload=_input_payload_from_messages(input_messages),
        )
    if text and isinstance(modified, str) and modified != text:
        response.content = modified


def _wrap_llm_with_guardrail(
    llm: BaseChatModel,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> BaseChatModel:
    """Wrap a ``BaseChatModel`` using Pydantic ``__class__`` swapping."""
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
                _apply_llm_post(
                    response,
                    evaluator,
                    action,
                    name,
                    input_messages=messages if isinstance(messages, list) else None,
                )
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
                _apply_llm_post(
                    response,
                    evaluator,
                    action,
                    name,
                    input_messages=messages if isinstance(messages, list) else None,
                )
            return response

    llm.__class__ = _GuardedLLM
    return llm


# ---------------------------------------------------------------------------
# StateGraph / CompiledStateGraph wrappers (AGENT scope)
# ---------------------------------------------------------------------------


def _apply_agent_input_guardrail(
    input: Any,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
) -> None:
    """Evaluate the last HumanMessage from agent input in-place.

    Every file the messages hold and every attachment a typed state key carries
    is sent with the text; files without text evaluate ``NO_TEXT_PLACEHOLDER``,
    which is never written back. An evaluator that drops files gets none, so
    files alone are not evaluated. An input with typed ``Attachment`` keys but no
    ``messages`` channel is not checked.
    """
    if not isinstance(input, dict) or "messages" not in input:
        return
    messages = input["messages"]
    if not isinstance(messages, list):
        return
    msg = _get_last_human_message(messages)
    text = _extract_message_text(msg) if msg is not None else ""
    attachments = (
        _agent_input_attachments(input) if _forwards_attachments(evaluator) else []
    )
    if not text and not attachments:
        return
    payload = text or NO_TEXT_PLACEHOLDER
    try:
        result = _evaluate(
            evaluator, payload, GuardrailExecutionStage.PRE, None, None, attachments
        )
    except Exception:
        logger.warning(_EVALUATION_FAILED_MSG, name, exc_info=True)
        return
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        modified = _run_action(
            action,
            result,
            payload,
            name,
            scope=GuardrailScope.AGENT,
            stage=GuardrailExecutionStage.PRE,
            component=component_label(GuardrailScope.AGENT),
        )
    if msg is not None and text and isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


def _apply_agent_output_guardrail(
    output: Any,
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
) -> None:
    """Evaluate the last AIMessage from agent output in-place.

    The held files the message or the structured response names, and the
    attachments the agent returns in typed state keys, are sent with the text;
    files without text evaluate ``NO_TEXT_PLACEHOLDER``, which is never written
    back. An evaluator that drops files gets none, so files alone are not
    evaluated. A typed input attachment the graph echoes in its output state is
    forwarded again here.
    """
    if not isinstance(output, dict) or "messages" not in output:
        return
    messages = output["messages"]
    if not isinstance(messages, list):
        return
    msg = _get_last_ai_message(messages)
    text = _extract_message_text(msg) if msg is not None else ""
    attachments = (
        _agent_output_attachments(output, msg)
        if _forwards_attachments(evaluator)
        else []
    )
    if not text and not attachments:
        return
    payload = text or NO_TEXT_PLACEHOLDER
    try:
        result = _evaluate(
            evaluator, payload, GuardrailExecutionStage.POST, None, None, attachments
        )
    except Exception:
        logger.warning(_EVALUATION_FAILED_MSG, name, exc_info=True)
        return
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        modified = _run_action(
            action,
            result,
            payload,
            name,
            scope=GuardrailScope.AGENT,
            stage=GuardrailExecutionStage.POST,
            component=component_label(GuardrailScope.AGENT),
            input_payload=_input_payload_from_messages(messages),
        )
    if msg is not None and text and isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


def _wrap_stategraph_with_guardrail(
    graph: "StateGraph[Any, Any]",
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> "StateGraph[Any, Any]":
    """Wrap a ``StateGraph``'s invoke/ainvoke to apply the guardrail."""
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
    graph: "CompiledStateGraph[Any, Any, Any]",
    evaluator: _EvaluatorFn,
    action: GuardrailAction,
    name: str,
    stage: GuardrailExecutionStage,
) -> "CompiledStateGraph[Any, Any, Any]":
    """Wrap a ``CompiledStateGraph``'s invoke/ainvoke to apply the guardrail."""
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


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------


class LangChainGuardrailAdapter:
    """Framework adapter for LangChain/LangGraph objects.

    Implements :class:`~uipath.platform.guardrails.decorators.GuardrailTargetAdapter`
    for ``BaseTool``, ``BaseChatModel``, ``StateGraph``, and
    ``CompiledStateGraph``.

    Auto-registered when ``uipath_langchain.guardrails`` is imported.
    """

    def recognize(self, target: Any) -> bool:
        """Return ``True`` if this adapter handles LangChain/LangGraph objects."""
        return isinstance(
            target, (BaseTool, BaseChatModel, StateGraph, CompiledStateGraph)
        )

    def wrap(
        self,
        target: Any,
        evaluator: _EvaluatorFn,
        action: GuardrailAction,
        name: str,
        stage: GuardrailExecutionStage,
    ) -> Any:
        """Wrap a LangChain/LangGraph object with guardrail enforcement."""
        if isinstance(target, BaseTool):
            return _wrap_tool_with_guardrail(target, evaluator, action, name, stage)
        if isinstance(target, BaseChatModel):
            return _wrap_llm_with_guardrail(target, evaluator, action, name, stage)
        if isinstance(target, CompiledStateGraph):
            return _wrap_compiled_graph_with_guardrail(
                target, evaluator, action, name, stage
            )
        if isinstance(target, StateGraph):
            return _wrap_stategraph_with_guardrail(
                target, evaluator, action, name, stage
            )
        return target
