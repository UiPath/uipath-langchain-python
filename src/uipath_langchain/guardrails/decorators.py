"""Guardrail decorators for applying guardrails directly to functions and objects."""

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailSelector,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform import UiPath
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    EnumListParameterValue,
    MapEnumParameterValue,
)

from .models import Entity, GuardrailAction


@dataclass
class GuardrailMetadata:
    """Metadata for a guardrail decorator.

    Args:
        guardrail_type: Type of guardrail ("pii", "prompt_injection", "deterministic")
        scope: Scope where guardrail applies (AGENT, LLM, TOOL)
        config: Type-specific configuration dictionary
        name: Name of the guardrail
        description: Optional description
        guardrail: The BuiltInValidatorGuardrail instance for evaluation
    """

    guardrail_type: str
    scope: GuardrailScope
    config: dict[str, Any]
    name: str
    description: str | None = None
    guardrail: BuiltInValidatorGuardrail | None = None


def _get_or_create_metadata_list(obj: Any) -> list[GuardrailMetadata]:
    """Get or create the guardrail metadata list on an object.

    Args:
        obj: Object to get/create metadata list for

    Returns:
        List of GuardrailMetadata instances
    """
    if not hasattr(obj, "_guardrail_metadata"):
        obj._guardrail_metadata = []
    return obj._guardrail_metadata


def _store_guardrail_metadata(obj: Any, metadata: GuardrailMetadata) -> None:
    """Store guardrail metadata on an object.

    Args:
        obj: Object to store metadata on
        metadata: GuardrailMetadata to store
    """
    metadata_list = _get_or_create_metadata_list(obj)
    metadata_list.append(metadata)


def _extract_guardrail_metadata(obj: Any) -> list[GuardrailMetadata]:
    """Extract all guardrail metadata from an object.

    Args:
        obj: Object to extract metadata from

    Returns:
        List of GuardrailMetadata instances, empty list if none found
    """
    if hasattr(obj, "_guardrail_metadata"):
        return list(obj._guardrail_metadata)
    return []


def _create_pii_guardrail(
    entities: Sequence[Entity],
    action: GuardrailAction,
    name: str,
    description: str | None,
    scope: GuardrailScope,
) -> BuiltInValidatorGuardrail:
    """Create a BuiltInValidatorGuardrail for PII detection.

    Args:
        entities: List of PII entities to detect
        action: Action to take when PII is detected
        name: Name of the guardrail
        description: Optional description
        scope: Scope where guardrail applies

    Returns:
        BuiltInValidatorGuardrail instance
    """
    # Extract entity names and thresholds
    entity_names = [entity.name for entity in entities]
    entity_thresholds = {entity.name: entity.threshold for entity in entities}

    # Create validator parameters
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

    # Create selector with scope
    selector_kwargs: dict[str, Any] = {"scopes": [scope]}

    return BuiltInValidatorGuardrail(
        id=str(uuid4()),
        name=name,
        description=description or f"Detects PII entities: {', '.join(entity_names)}",
        enabled_for_evals=True,
        selector=GuardrailSelector(**selector_kwargs),
        guardrail_type="builtInValidator",
        validator_type="pii_detection",
        validator_parameters=validator_parameters,
    )


def _extract_text_from_messages(messages: list[BaseMessage]) -> str:
    """Extract text content from messages."""
    text_parts = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                # Handle multimodal content
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
    return "\n".join(text_parts)


def _detect_scope(obj: Any) -> GuardrailScope:
    """Detect the guardrail scope from an object.

    Args:
        obj: Object to detect scope for

    Returns:
        GuardrailScope (LLM or AGENT)
    """
    # Check if it's an LLM instance
    if isinstance(obj, BaseChatModel):
        return GuardrailScope.LLM

    # Check if it's a StateGraph
    if isinstance(obj, StateGraph):
        return GuardrailScope.AGENT

    # Check if it's a function that returns StateGraph
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        sig = inspect.signature(obj)
        # Check return annotation
        if sig.return_annotation != inspect.Signature.empty:
            if sig.return_annotation == StateGraph or (
                hasattr(sig.return_annotation, "__origin__")
                and sig.return_annotation.__origin__ is StateGraph
            ):
                return GuardrailScope.AGENT

    # Default to AGENT for functions (assumed to be agent creation functions)
    if callable(obj):
        return GuardrailScope.AGENT

    raise ValueError(
        f"Cannot determine scope for object of type {type(obj)}. "
        "Object must be a BaseChatModel, StateGraph, or a function."
    )


def _evaluate_pii_guardrail(
    data: str | dict[str, Any],
    guardrail: BuiltInValidatorGuardrail,
    uipath: UiPath,
) -> GuardrailValidationResult:
    """Evaluate a PII guardrail against data.

    Args:
        data: Data to evaluate (string or dict)
        guardrail: BuiltInValidatorGuardrail instance
        uipath: UiPath instance for guardrails service

    Returns:
        GuardrailValidationResult
    """
    return uipath.guardrails.evaluate_guardrail(data, guardrail)


def _handle_guardrail_result(
    result: GuardrailValidationResult,
    data: str | dict[str, Any],
    action: GuardrailAction,
    guardrail_name: str,
) -> str | dict[str, Any] | None:
    """Handle guardrail validation result using action.

    Args:
        result: GuardrailValidationResult from evaluation
        data: Original data that was validated
        action: GuardrailAction to process the result
        guardrail_name: Name of the guardrail

    Returns:
        Modified data or None if no modification needed
    """
    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
        return action.handle_validation_result(result, data, guardrail_name)
    return None


def pii_guardrail(
    func: Callable[..., Any] | BaseChatModel | None = None,
    *,
    entities: Sequence[Entity] | None = None,
    action: GuardrailAction | None = None,
    name: str = "PII Detection",
    description: str | None = None,
):
    """Decorator for PII detection guardrails.

    Can be applied to LLM instances or agent functions/graphs.
    Scope is automatically detected from the decorated object.

    Args:
        func: Function or object to decorate (when used without parentheses)
        entities: List of PII entities to detect (required when used with parentheses)
        action: Action to take when PII is detected (required when used with parentheses)
        name: Optional name for the guardrail
        description: Optional description for the guardrail

    Example:
        ```python
        # Apply to LLM instance directly
        llm = pii_guardrail(
            entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
            action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
        )(UiPathChat(model="gpt-4o"))

        # Apply to agent function
        @pii_guardrail(
            entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
            action=LogAction(severity_level=AgentGuardrailSeverityLevel.WARNING),
        )
        def create_agent():
            return create_agent(...)
        ```

    Returns:
        Decorated function or object
    """
    # Handle both @pii_guardrail and @pii_guardrail(...) patterns
    if func is None:
        # Called as @pii_guardrail(...) or pii_guardrail(...)(obj) - return decorator
        def decorator(
            f: Callable[..., Any] | BaseChatModel,
        ) -> Callable[..., Any] | BaseChatModel:
            return _apply_pii_guardrail(f, entities, action, name, description)

        return decorator
    else:
        # Called as @pii_guardrail - apply decorator directly
        # In this case, entities and action must be provided via metadata
        # This pattern is less common, but we support it
        if entities is None or action is None:
            raise ValueError(
                "When using @pii_guardrail without parentheses, "
                "you must provide entities and action as keyword arguments."
            )
        return _apply_pii_guardrail(func, entities, action, name, description)


def _apply_pii_guardrail(
    obj: Callable[..., Any] | BaseChatModel,
    entities: Sequence[Entity] | None,
    action: GuardrailAction | None,
    name: str,
    description: str | None,
) -> Callable[..., Any] | BaseChatModel:
    """Apply PII guardrail to an object.

    Args:
        obj: Object to apply guardrail to
        entities: List of PII entities to detect
        action: Action to take when PII is detected
        name: Name of the guardrail
        description: Optional description

    Returns:
        Decorated object
    """
    if entities is None or not entities:
        raise ValueError("entities must be provided and non-empty")
    if action is None:
        raise ValueError("action must be provided")
    if not isinstance(action, GuardrailAction):
        raise ValueError("action must be an instance of GuardrailAction")

    # Detect scope
    scope = _detect_scope(obj)

    # Create guardrail instance
    guardrail = _create_pii_guardrail(entities, action, name, description, scope)

    # Create metadata
    metadata = GuardrailMetadata(
        guardrail_type="pii",
        scope=scope,
        config={"entities": list(entities), "action": action},
        name=name,
        description=description,
        guardrail=guardrail,
    )

    # Store metadata
    _store_guardrail_metadata(obj, metadata)

    # Wrap based on type
    if isinstance(obj, BaseChatModel):
        return _wrap_llm_with_guardrail(obj, metadata)
    elif callable(obj):
        return _wrap_function_with_guardrail(obj, metadata)
    else:
        # For other types, just store metadata
        return obj


def _wrap_llm_with_guardrail(
    llm: BaseChatModel, metadata: GuardrailMetadata
) -> BaseChatModel:
    """Wrap LLM methods to apply guardrails.

    Args:
        llm: LLM instance to wrap
        metadata: GuardrailMetadata with guardrail configuration

    Returns:
        Wrapped LLM instance
    """
    guardrail = metadata.guardrail
    action = metadata.config["action"]
    guardrail_name = metadata.name
    uipath = UiPath()

    # Store original methods
    original_invoke = llm.invoke
    original_ainvoke = llm.ainvoke

    @wraps(original_invoke)
    def wrapped_invoke(messages, config=None, **kwargs):
        """Wrap invoke to check messages before LLM call."""
        if isinstance(messages, list):
            # Extract text from messages
            text = _extract_text_from_messages(messages)

            if text:
                # Evaluate guardrail
                result = _evaluate_pii_guardrail(text, guardrail, uipath)

                # Handle result
                modified_text = _handle_guardrail_result(
                    result, text, action, guardrail_name
                )

                # Apply modifications to messages if needed
                if modified_text and modified_text != text:
                    # Update message content
                    for msg in messages:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            if isinstance(msg.content, str) and text in msg.content:
                                msg.content = msg.content.replace(
                                    text, modified_text, 1
                                )
                                break

        # Call original LLM
        response = original_invoke(messages, config, **kwargs)

        # Optionally check output
        if isinstance(response, AIMessage) and isinstance(response.content, str):
            output_text = response.content
            if output_text:
                result = _evaluate_pii_guardrail(output_text, guardrail, uipath)
                modified_output = _handle_guardrail_result(
                    result, output_text, action, guardrail_name
                )
                if modified_output and modified_output != output_text:
                    response.content = modified_output

        return response

    @wraps(original_ainvoke)
    async def wrapped_ainvoke(messages, config=None, **kwargs):
        """Wrap ainvoke to check messages before LLM call."""
        if isinstance(messages, list):
            # Extract text from messages
            text = _extract_text_from_messages(messages)

            if text:
                # Evaluate guardrail
                result = _evaluate_pii_guardrail(text, guardrail, uipath)

                # Handle result
                modified_text = _handle_guardrail_result(
                    result, text, action, guardrail_name
                )

                # Apply modifications to messages if needed
                if modified_text and modified_text != text:
                    # Update message content
                    for msg in messages:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            if isinstance(msg.content, str) and text in msg.content:
                                msg.content = msg.content.replace(
                                    text, modified_text, 1
                                )
                                break

        # Call original LLM
        response = await original_ainvoke(messages, config, **kwargs)

        # Optionally check output
        if isinstance(response, AIMessage) and isinstance(response.content, str):
            output_text = response.content
            if output_text:
                result = _evaluate_pii_guardrail(output_text, guardrail, uipath)
                modified_output = _handle_guardrail_result(
                    result, output_text, action, guardrail_name
                )
                if modified_output and modified_output != output_text:
                    response.content = modified_output

        return response

    # Replace methods
    llm.invoke = wrapped_invoke
    llm.ainvoke = wrapped_ainvoke

    return llm


def _wrap_stategraph_with_guardrail(
    graph: StateGraph, metadata: GuardrailMetadata
) -> StateGraph:
    """Wrap StateGraph methods to apply guardrails.

    Args:
        graph: StateGraph instance to wrap
        metadata: GuardrailMetadata with guardrail configuration

    Returns:
        Wrapped StateGraph instance
    """
    guardrail = metadata.guardrail
    action = metadata.config["action"]
    guardrail_name = metadata.name
    uipath = UiPath()

    # Store original methods if they exist
    if hasattr(graph, "invoke"):
        original_invoke = graph.invoke

        @wraps(original_invoke)
        def wrapped_invoke(input, config=None, **kwargs):
            """Wrap invoke to check state before agent execution."""
            if isinstance(input, dict) and "messages" in input:
                messages = input["messages"]
                if isinstance(messages, list):
                    text = _extract_text_from_messages(messages)

                    if text:
                        # Evaluate guardrail
                        result = _evaluate_pii_guardrail(text, guardrail, uipath)

                        # Handle result
                        modified_text = _handle_guardrail_result(
                            result, text, action, guardrail_name
                        )

                        # Apply modifications to input if needed
                        if modified_text and modified_text != text:
                            # Update message content
                            for msg in messages:
                                if isinstance(msg, (HumanMessage, AIMessage)):
                                    if (
                                        isinstance(msg.content, str)
                                        and text in msg.content
                                    ):
                                        msg.content = msg.content.replace(
                                            text, modified_text, 1
                                        )
                                        break

            # Call original invoke
            output = original_invoke(input, config, **kwargs)

            # Check output
            if isinstance(output, dict) and "messages" in output:
                output_messages = output["messages"]
                if isinstance(output_messages, list):
                    output_text = _extract_text_from_messages(output_messages)
                    if output_text:
                        output_result = _evaluate_pii_guardrail(
                            output_text, guardrail, uipath
                        )
                        modified_output = _handle_guardrail_result(
                            output_result, output_text, action, guardrail_name
                        )
                        if modified_output and modified_output != output_text:
                            # Update output messages
                            for msg in output_messages:
                                if isinstance(msg, (HumanMessage, AIMessage)):
                                    if (
                                        isinstance(msg.content, str)
                                        and output_text in msg.content
                                    ):
                                        msg.content = msg.content.replace(
                                            output_text, modified_output, 1
                                        )
                                        break

            return output

        graph.invoke = wrapped_invoke

    if hasattr(graph, "ainvoke"):
        original_ainvoke = graph.ainvoke

        @wraps(original_ainvoke)
        async def wrapped_ainvoke(input, config=None, **kwargs):
            """Wrap ainvoke to check state before agent execution."""
            if isinstance(input, dict) and "messages" in input:
                messages = input["messages"]
                if isinstance(messages, list):
                    text = _extract_text_from_messages(messages)

                    if text:
                        # Evaluate guardrail
                        result = _evaluate_pii_guardrail(text, guardrail, uipath)

                        # Handle result
                        modified_text = _handle_guardrail_result(
                            result, text, action, guardrail_name
                        )

                        # Apply modifications to input if needed
                        if modified_text and modified_text != text:
                            # Update message content
                            for msg in messages:
                                if isinstance(msg, (HumanMessage, AIMessage)):
                                    if (
                                        isinstance(msg.content, str)
                                        and text in msg.content
                                    ):
                                        msg.content = msg.content.replace(
                                            text, modified_text, 1
                                        )
                                        break

            # Call original ainvoke
            output = await original_ainvoke(input, config, **kwargs)

            # Check output
            if isinstance(output, dict) and "messages" in output:
                output_messages = output["messages"]
                if isinstance(output_messages, list):
                    output_text = _extract_text_from_messages(output_messages)
                    if output_text:
                        output_result = _evaluate_pii_guardrail(
                            output_text, guardrail, uipath
                        )
                        modified_output = _handle_guardrail_result(
                            output_result, output_text, action, guardrail_name
                        )
                        if modified_output and modified_output != output_text:
                            # Update output messages
                            for msg in output_messages:
                                if isinstance(msg, (HumanMessage, AIMessage)):
                                    if (
                                        isinstance(msg.content, str)
                                        and output_text in msg.content
                                    ):
                                        msg.content = msg.content.replace(
                                            output_text, modified_output, 1
                                        )
                                        break

            return output

        graph.ainvoke = wrapped_ainvoke

    return graph


def _wrap_function_with_guardrail(
    func: Callable[..., Any], metadata: GuardrailMetadata
) -> Callable[..., Any]:
    """Wrap a function to apply guardrails.

    Args:
        func: Function to wrap
        metadata: GuardrailMetadata with guardrail configuration

    Returns:
        Wrapped function
    """
    guardrail = metadata.guardrail
    action = metadata.config["action"]
    guardrail_name = metadata.name
    uipath = UiPath()

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        """Wrap function to check state/messages before execution."""
        # Call original function
        result = func(*args, **kwargs)

        # If result is a StateGraph, wrap its methods
        if isinstance(result, StateGraph):
            return _wrap_stategraph_with_guardrail(result, metadata)

        # For other results, check if they have messages
        if isinstance(result, dict) and "messages" in result:
            output_messages = result["messages"]
            if isinstance(output_messages, list):
                output_text = _extract_text_from_messages(output_messages)
                if output_text:
                    output_result = _evaluate_pii_guardrail(
                        output_text, guardrail, uipath
                    )
                    modified_output = _handle_guardrail_result(
                        output_result, output_text, action, guardrail_name
                    )
                    if modified_output and modified_output != output_text:
                        # Update output messages
                        for msg in output_messages:
                            if isinstance(msg, (HumanMessage, AIMessage)):
                                if (
                                    isinstance(msg.content, str)
                                    and output_text in msg.content
                                ):
                                    msg.content = msg.content.replace(
                                        output_text, modified_output, 1
                                    )
                                    break

        return result

    @wraps(func)
    async def wrapped_async_func(*args, **kwargs):
        """Wrap async function to check state/messages before execution."""
        # Call original function
        result = await func(*args, **kwargs)

        # If result is a StateGraph, wrap its methods
        if isinstance(result, StateGraph):
            return _wrap_stategraph_with_guardrail(result, metadata)

        # For other results, check if they have messages
        if isinstance(result, dict) and "messages" in result:
            output_messages = result["messages"]
            if isinstance(output_messages, list):
                output_text = _extract_text_from_messages(output_messages)
                if output_text:
                    output_result = _evaluate_pii_guardrail(
                        output_text, guardrail, uipath
                    )
                    modified_output = _handle_guardrail_result(
                        output_result, output_text, action, guardrail_name
                    )
                    if modified_output and modified_output != output_text:
                        # Update output messages
                        for msg in output_messages:
                            if isinstance(msg, (HumanMessage, AIMessage)):
                                if (
                                    isinstance(msg.content, str)
                                    and output_text in msg.content
                                ):
                                    msg.content = msg.content.replace(
                                        output_text, modified_output, 1
                                    )
                                    break

        return result

    # Return appropriate wrapper based on whether function is async
    if inspect.iscoroutinefunction(func):
        return wrapped_async_func
    else:
        return wrapped_func
