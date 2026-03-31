"""Shared base utilities for guardrail decorators."""

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

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
from uipath.platform import UiPath
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from ..enums import GuardrailExecutionStage
from ..models import GuardrailAction


@dataclass
class GuardrailMetadata:
    """Metadata for a guardrail decorator.

    Args:
        guardrail_type: Type of guardrail ("pii", "prompt_injection", "deterministic")
        scope: Scope where guardrail applies (AGENT, LLM, TOOL)
        config: Type-specific configuration dictionary
        name: Name of the guardrail
        description: Optional description
        guardrail: The BuiltInValidatorGuardrail instance for API-based evaluation
        wrap_tool: Optional callable that wraps a BaseTool with this guardrail's logic.
            Set by each decorator so that _wrap_function_with_guardrail can delegate
            tool wrapping without knowing the concrete guardrail type.
    """

    guardrail_type: str
    scope: GuardrailScope
    config: dict[str, Any]
    name: str
    description: str | None = None
    guardrail: BuiltInValidatorGuardrail | None = None
    wrap_tool: Callable[["BaseTool", "GuardrailMetadata"], "BaseTool"] | None = None
    wrap_llm: (
        Callable[["BaseChatModel", "GuardrailMetadata"], "BaseChatModel"] | None
    ) = None


def _get_or_create_metadata_list(obj: Any) -> list[GuardrailMetadata]:
    """Get or create the guardrail metadata list on an object."""
    if not hasattr(obj, "_guardrail_metadata"):
        obj._guardrail_metadata = []
    return obj._guardrail_metadata


def _store_guardrail_metadata(obj: Any, metadata: GuardrailMetadata) -> None:
    """Store guardrail metadata on an object."""
    metadata_list = _get_or_create_metadata_list(obj)
    metadata_list.append(metadata)


def _extract_guardrail_metadata(obj: Any) -> list[GuardrailMetadata]:
    """Extract all guardrail metadata from an object."""
    if hasattr(obj, "_guardrail_metadata"):
        return list(obj._guardrail_metadata)
    return []


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
    """Extract text content from a single message."""
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
    """Apply a modified text string back to a message in-place.

    For str content, replaces it directly. For multimodal list content,
    replaces the first text part.
    """
    if isinstance(msg.content, str):
        msg.content = modified
    elif isinstance(msg.content, list):
        for part in msg.content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = modified
                break


def _detect_scope(obj: Any) -> GuardrailScope:
    """Detect the guardrail scope from an object.

    Returns:
        GuardrailScope.TOOL for BaseTool instances.
        GuardrailScope.LLM for BaseChatModel instances.
        GuardrailScope.AGENT for StateGraph or CompiledStateGraph instances,
            including subgraphs — guardrails apply at the boundary of any graph
            execution, not only the top-level agent.
        GuardrailScope.AGENT for plain functions/methods (agent factory functions),
            optionally annotated with a StateGraph or CompiledStateGraph return type.
    """
    if isinstance(obj, BaseTool):
        return GuardrailScope.TOOL

    if isinstance(obj, BaseChatModel):
        return GuardrailScope.LLM

    if isinstance(obj, StateGraph):
        return GuardrailScope.AGENT

    if isinstance(obj, CompiledStateGraph):
        return GuardrailScope.AGENT

    if inspect.isfunction(obj) or inspect.ismethod(obj):
        sig = inspect.signature(obj)
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation in (StateGraph, CompiledStateGraph) or (
                hasattr(sig.return_annotation, "__origin__")
                and sig.return_annotation.__origin__ in (StateGraph, CompiledStateGraph)
            ):
                return GuardrailScope.AGENT
        return GuardrailScope.AGENT

    raise ValueError(
        f"Cannot determine scope for object of type {type(obj)}. "
        "Object must be a BaseTool, BaseChatModel, StateGraph, CompiledStateGraph, "
        "or a callable function/method (agent factory)."
    )


def _evaluate_guardrail(
    data: str | dict[str, Any],
    guardrail: BuiltInValidatorGuardrail,
    uipath: UiPath,
) -> GuardrailValidationResult:
    """Evaluate a guardrail against data via the UiPath API."""
    return uipath.guardrails.evaluate_guardrail(data, guardrail)


def _handle_guardrail_result(
    result: GuardrailValidationResult,
    data: str | dict[str, Any],
    action: GuardrailAction,
    guardrail_name: str,
) -> str | dict[str, Any] | None:
    """Handle guardrail validation result using action."""
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        return action.handle_validation_result(result, data, guardrail_name)
    return None


def _evaluate_rules(
    rules: Sequence[Callable[..., bool]],
    stage: GuardrailExecutionStage,
    input_data: dict[str, Any] | None,
    output_data: dict[str, Any] | None,
    guardrail_name: str = "Rule",
) -> GuardrailValidationResult:
    """Evaluate deterministic rules and return a validation result.

    All rules must detect violations to trigger. If any rule passes (returns False),
    the guardrail passes. Empty rules always trigger the action.
    """
    import logging

    logger = logging.getLogger(__name__)

    if not rules:
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="Empty rules — always apply action",
        )

    violations: list[str] = []
    passed_rules: list[str] = []
    evaluated_count = 0

    for rule in rules:
        try:
            sig = inspect.signature(rule)
            param_count = len(sig.parameters)

            if stage == GuardrailExecutionStage.PRE:
                if input_data is None or param_count != 1:
                    continue
                violation = rule(input_data)
                evaluated_count += 1
            else:
                if output_data is None:
                    continue
                if param_count == 2 and input_data is not None:
                    violation = rule(input_data, output_data)
                elif param_count == 1:
                    violation = rule(output_data)
                else:
                    continue
                evaluated_count += 1

            if violation:
                violations.append(f"Rule {guardrail_name} detected violation")
            else:
                passed_rules.append(f"Rule {guardrail_name}")
        except Exception as e:
            logger.error(f"Error in rule function {guardrail_name}: {e}", exc_info=True)
            violations.append(f"Rule {guardrail_name} raised exception: {str(e)}")
            evaluated_count += 1

    if evaluated_count == 0:
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED,
            reason="No applicable rules to evaluate",
        )

    if passed_rules:
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED,
            reason=f"Rules passed: {', '.join(passed_rules)}",
        )

    return GuardrailValidationResult(
        result=GuardrailValidationResultType.VALIDATION_FAILED,
        reason="; ".join(violations),
    )


# ---------------------------------------------------------------------------
# Module-level tool I/O helpers shared by PII and deterministic tool wrappers
# ---------------------------------------------------------------------------


def _is_tool_call_envelope(tool_input: Any) -> bool:
    """Return True if tool_input is a LangGraph tool-call envelope dict."""
    return (
        isinstance(tool_input, dict)
        and "args" in tool_input
        and tool_input.get("type") == "tool_call"
    )


def _extract_input(tool_input: Any) -> dict[str, Any]:
    """Normalise tool input to a dict for rule/guardrail evaluation.

    LangGraph passes the raw tool-call dict ({"name": ..., "args": {...}, "id": ...,
    "type": "tool_call"}) to tool.invoke/ainvoke. Unwrap "args" so rules can access
    the actual tool arguments (e.g. args.get("joke", "")) directly.
    """
    if _is_tool_call_envelope(tool_input):
        args = tool_input["args"]
        if isinstance(args, dict):
            return args
    if isinstance(tool_input, dict):
        return tool_input
    return {"input": tool_input}


def _rewrap_input(original_tool_input: Any, modified_args: dict[str, Any]) -> Any:
    """Re-wrap modified args back into the original tool-call envelope (if applicable)."""
    if _is_tool_call_envelope(original_tool_input):
        import copy

        wrapped = copy.copy(original_tool_input)
        wrapped["args"] = modified_args
        return wrapped
    return modified_args


def _extract_output(result: Any) -> dict[str, Any]:
    """Normalise tool output to a dict for guardrail/rule evaluation.

    Handles ToolMessage and Command (returned when the tool is called through
    LangGraph's tool node) by extracting their string content first, then
    parsing as JSON/literal-eval. Falls back to {"output": content} for
    plain strings and {"output": result} for anything else.
    """
    import ast
    import json

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

    if isinstance(content, dict):
        return content
    if isinstance(result, dict):
        return result
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"output": parsed}
        except ValueError:
            try:
                parsed = ast.literal_eval(content)
                return parsed if isinstance(parsed, dict) else {"output": parsed}
            except (ValueError, SyntaxError):
                return {"output": content}
    return {"output": content}


# ---------------------------------------------------------------------------
# Module-level LLM guardrail helpers shared by PII and prompt-injection wrappers
# ---------------------------------------------------------------------------


def _apply_llm_input_guardrail(
    messages: list[BaseMessage],
    guardrail: BuiltInValidatorGuardrail,
    uipath: UiPath,
    action: GuardrailAction,
    guardrail_name: str,
) -> None:
    """Evaluate a guardrail against the last HumanMessage (PRE stage).

    Only the most recent user input is evaluated — prior turns were already
    evaluated in previous invocations. Modifies the message content in-place
    if the action returns a replacement string.
    """
    msg = _get_last_human_message(messages)
    if msg is None:
        return
    text = _extract_message_text(msg)
    if not text:
        return
    try:
        eval_result = _evaluate_guardrail(text, guardrail, uipath)
    except Exception:
        return
    modified = _handle_guardrail_result(eval_result, text, action, guardrail_name)
    if isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


def _apply_llm_output_guardrail(
    response: AIMessage,
    guardrail: BuiltInValidatorGuardrail,
    uipath: UiPath,
    action: GuardrailAction,
    guardrail_name: str,
) -> None:
    """Evaluate a guardrail against the LLM response text (POST stage).

    Modifies ``response.content`` in-place if the action returns a replacement string.
    """
    if not isinstance(response.content, str) or not response.content:
        return
    try:
        eval_result = _evaluate_guardrail(response.content, guardrail, uipath)
    except Exception:
        return
    modified = _handle_guardrail_result(
        eval_result, response.content, action, guardrail_name
    )
    if isinstance(modified, str) and modified != response.content:
        response.content = modified


def _apply_guardrail_to_message_list(
    messages: list[BaseMessage],
    guardrail: BuiltInValidatorGuardrail,
    uipath: UiPath,
    action: GuardrailAction,
    guardrail_name: str,
    target_type: type[BaseMessage] = HumanMessage,
) -> None:
    """Evaluate a guardrail against the last message of target_type and modify it in-place.

    Pass target_type=HumanMessage (default) for PRE/input evaluation,
    or target_type=AIMessage for POST/output evaluation.
    """
    msg: BaseMessage | None = None
    for m in reversed(messages):
        if isinstance(m, target_type):
            msg = m
            break
    if msg is None:
        return
    text = _extract_message_text(msg)
    if not text:
        return
    try:
        result = _evaluate_guardrail(text, guardrail, uipath)
    except Exception:
        return
    modified = _handle_guardrail_result(result, text, action, guardrail_name)
    if isinstance(modified, str) and modified != text:
        _apply_message_text_modification(msg, modified)


def _apply_guardrail_to_input_messages(
    input_data: Any,
    guardrail: BuiltInValidatorGuardrail,
    uipath: UiPath,
    action: GuardrailAction,
    guardrail_name: str,
) -> None:
    """If input is a dict with a 'messages' list, apply guardrail to it in-place."""
    if not isinstance(input_data, dict) or "messages" not in input_data:
        return
    messages = input_data["messages"]
    if not isinstance(messages, list):
        return
    _apply_guardrail_to_message_list(
        messages, guardrail, uipath, action, guardrail_name
    )


def _apply_guardrail_to_output_messages(
    output: Any,
    guardrail: BuiltInValidatorGuardrail,
    uipath: UiPath,
    action: GuardrailAction,
    guardrail_name: str,
) -> None:
    """If output is a dict with a 'messages' list, apply guardrail to the last AIMessage in-place."""
    if not isinstance(output, dict) or "messages" not in output:
        return
    messages = output["messages"]
    if not isinstance(messages, list):
        return
    _apply_guardrail_to_message_list(
        messages, guardrail, uipath, action, guardrail_name, target_type=AIMessage
    )


def _wrap_stategraph_with_guardrail(
    graph: StateGraph[Any, Any], metadata: GuardrailMetadata
) -> StateGraph[Any, Any]:
    """Wrap StateGraph invoke/ainvoke to apply guardrails."""
    built_in_guardrail = metadata.guardrail
    if built_in_guardrail is None:
        return graph
    action = metadata.config["action"]
    guardrail_name = metadata.name
    uipath = UiPath()

    if hasattr(graph, "invoke"):
        original_invoke = graph.invoke

        @wraps(original_invoke)
        def wrapped_invoke(input, config=None, **kwargs):
            _apply_guardrail_to_input_messages(
                input, built_in_guardrail, uipath, action, guardrail_name
            )
            output = original_invoke(input, config, **kwargs)
            _apply_guardrail_to_output_messages(
                output, built_in_guardrail, uipath, action, guardrail_name
            )
            return output

        graph.invoke = wrapped_invoke

    if hasattr(graph, "ainvoke"):
        original_ainvoke = graph.ainvoke

        @wraps(original_ainvoke)
        async def wrapped_ainvoke(input, config=None, **kwargs):
            _apply_guardrail_to_input_messages(
                input, built_in_guardrail, uipath, action, guardrail_name
            )
            output = await original_ainvoke(input, config, **kwargs)
            _apply_guardrail_to_output_messages(
                output, built_in_guardrail, uipath, action, guardrail_name
            )
            return output

        graph.ainvoke = wrapped_ainvoke

    return graph


def _wrap_compiled_graph_with_guardrail(
    graph: CompiledStateGraph[Any, Any, Any], metadata: GuardrailMetadata
) -> CompiledStateGraph[Any, Any, Any]:
    """Wrap a CompiledStateGraph's invoke/ainvoke to apply guardrails."""
    built_in_guardrail = metadata.guardrail
    if built_in_guardrail is None:
        return graph
    action = metadata.config["action"]
    guardrail_name = metadata.name
    uipath = UiPath()

    original_invoke = graph.invoke
    original_ainvoke = graph.ainvoke

    @wraps(original_invoke)
    def wrapped_invoke(input, config=None, **kwargs):
        _apply_guardrail_to_input_messages(
            input, built_in_guardrail, uipath, action, guardrail_name
        )
        output = original_invoke(input, config, **kwargs)
        _apply_guardrail_to_output_messages(
            output, built_in_guardrail, uipath, action, guardrail_name
        )
        return output

    @wraps(original_ainvoke)
    async def wrapped_ainvoke(input, config=None, **kwargs):
        _apply_guardrail_to_input_messages(
            input, built_in_guardrail, uipath, action, guardrail_name
        )
        output = await original_ainvoke(input, config, **kwargs)
        _apply_guardrail_to_output_messages(
            output, built_in_guardrail, uipath, action, guardrail_name
        )
        return output

    graph.invoke = wrapped_invoke  # type: ignore[method-assign]
    graph.ainvoke = wrapped_ainvoke  # type: ignore[method-assign]
    return graph


def _wrap_function_with_guardrail(
    func: Callable[..., Any], metadata: GuardrailMetadata
) -> Callable[..., Any]:
    """Wrap a function to apply guardrails.

    After calling the function, inspects the return value:
    - StateGraph / CompiledStateGraph: delegates to the appropriate graph wrapper
    - BaseChatModel: delegates to the LLM wrapper
    - BaseTool: delegates to the tool wrapper
    """
    built_in_guardrail = metadata.guardrail
    if built_in_guardrail is None:
        return func
    action = metadata.config["action"]
    guardrail_name = metadata.name
    uipath = UiPath()

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, StateGraph):
            return _wrap_stategraph_with_guardrail(result, metadata)
        if isinstance(result, CompiledStateGraph):
            return _wrap_compiled_graph_with_guardrail(result, metadata)
        if isinstance(result, BaseChatModel):
            if metadata.wrap_llm is not None:
                return metadata.wrap_llm(result, metadata)
        if isinstance(result, BaseTool) and metadata.wrap_tool is not None:
            return metadata.wrap_tool(result, metadata)
        _apply_guardrail_to_output_messages(
            result, built_in_guardrail, uipath, action, guardrail_name
        )
        return result

    @wraps(func)
    async def wrapped_async_func(*args, **kwargs):
        result = await func(*args, **kwargs)
        if isinstance(result, StateGraph):
            return _wrap_stategraph_with_guardrail(result, metadata)
        if isinstance(result, CompiledStateGraph):
            return _wrap_compiled_graph_with_guardrail(result, metadata)
        if isinstance(result, BaseChatModel):
            if metadata.wrap_llm is not None:
                return metadata.wrap_llm(result, metadata)
        if isinstance(result, BaseTool) and metadata.wrap_tool is not None:
            return metadata.wrap_tool(result, metadata)
        _apply_guardrail_to_output_messages(
            result, built_in_guardrail, uipath, action, guardrail_name
        )
        return result

    if inspect.iscoroutinefunction(func):
        return wrapped_async_func
    return wrapped_func
