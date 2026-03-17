"""LangChain/LangGraph adapter for the ``@guardrail`` decorator.

Registers a :class:`LangChainGuardrailAdapter` that teaches the platform
``@guardrail`` decorator how to wrap LangChain and LangGraph objects:

- ``BaseTool`` → TOOL scope
- ``BaseChatModel`` → LLM scope
- ``StateGraph`` / ``CompiledStateGraph`` → AGENT scope

The adapter is auto-registered when ``uipath_langchain.guardrails`` is imported.
"""

import logging
from functools import wraps
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from uipath.core.guardrails import GuardrailValidationResultType
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
from uipath_langchain.guardrails.middlewares._utils import create_modified_tool_result

logger = logging.getLogger(__name__)


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
    """Wrap a ``BaseTool`` using Pydantic ``__class__`` swapping."""
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
        modified = None
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            try:
                modified = action.handle_validation_result(result, input_data, name)
            except GuardrailBlockException as exc:
                raise _convert_block_exception(exc) from exc
        if modified is not None and isinstance(modified, dict):
            return _rewrap_input(tool_input, modified)
        return tool_input

    def _apply_post(tool_input: Any, raw_result: Any) -> Any:
        input_data = _extract_input(tool_input)
        output_data = _lc_extract_output(raw_result)
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
        modified = None
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            try:
                modified = action.handle_validation_result(result, output_data, name)
            except GuardrailBlockException as exc:
                raise _convert_block_exception(exc) from exc
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
        # guardrail to fire twice: once inside self.invoke and once after
        # super().ainvoke() returns.

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
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        try:
            modified = action.handle_validation_result(result, text, name)
        except GuardrailBlockException as exc:
            raise _convert_block_exception(exc) from exc
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
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        try:
            modified = action.handle_validation_result(result, response.content, name)
        except GuardrailBlockException as exc:
            raise _convert_block_exception(exc) from exc
    if isinstance(modified, str) and modified != response.content:
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


# ---------------------------------------------------------------------------
# StateGraph / CompiledStateGraph wrappers (AGENT scope)
# ---------------------------------------------------------------------------


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
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        try:
            modified = action.handle_validation_result(result, text, name)
        except GuardrailBlockException as exc:
            raise _convert_block_exception(exc) from exc
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
    modified = None
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        try:
            modified = action.handle_validation_result(result, text, name)
        except GuardrailBlockException as exc:
            raise _convert_block_exception(exc) from exc
    if isinstance(modified, str) and modified != text:
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
