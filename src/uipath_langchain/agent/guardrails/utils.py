import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.react.types import AgentGuardrailsGraphState
from uipath_langchain.agent.tools.utils import sanitize_tool_name

logger = logging.getLogger(__name__)


def _extract_tool_args_from_message(
    message: AnyMessage, tool_name: str
) -> dict[str, Any]:
    """Extract tool call arguments from an AIMessage.

    Args:
        message: The message to extract from.
        tool_name: Name of the tool to extract arguments from.

    Returns:
        Dict containing tool call arguments, or empty dict if not found.
    """
    if not isinstance(message, AIMessage):
        return {}

    if not message.tool_calls:
        return {}

    # Find the first tool call with matching name
    for tool_call in message.tool_calls:
        call_name = (
            tool_call.get("name")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "name", None)
        )
        if call_name == tool_name:
            # Extract args from the tool call
            args = (
                tool_call.get("args")
                if isinstance(tool_call, dict)
                else getattr(tool_call, "args", None)
            )
            if args is not None:
                # Args should already be a dict
                if isinstance(args, dict):
                    return args
                # If it's a JSON string, parse it
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse tool args as JSON for tool '%s': %s",
                            tool_name,
                            args[:100] if len(args) > 100 else args,
                        )
                        return {}

    return {}


def _extract_tool_input_data(
    state: AgentGuardrailsGraphState, tool_name: str, execution_stage: ExecutionStage
) -> dict[str, Any]:
    """Extract tool call arguments as dict for deterministic guardrails.

    Args:
        state: The current agent graph state.
        tool_name: Name of the tool to extract arguments from.
        execution_stage: PRE_EXECUTION or POST_EXECUTION.

    Returns:
        Dict containing tool call arguments, or empty dict if not found.
        - For PRE_EXECUTION: extracts from last message
        - For POST_EXECUTION: extracts from second-to-last message
    """
    if not state.messages:
        return {}

    # For PRE_EXECUTION, look at last message
    # For POST_EXECUTION, look at second-to-last message (before the ToolMessage)
    if execution_stage == ExecutionStage.PRE_EXECUTION:
        if len(state.messages) < 1:
            return {}
        message = state.messages[-1]
    else:  # POST_EXECUTION
        if len(state.messages) < 2:
            return {}
        message = state.messages[-2]

    return _extract_tool_args_from_message(message, tool_name)


def _extract_tool_output_data(state: AgentGuardrailsGraphState) -> dict[str, Any]:
    """Extract tool execution output as dict for POST_EXECUTION deterministic guardrails.

    Args:
        state: The current agent graph state.

    Returns:
        Dict containing tool output. If output is not valid JSON, wraps it in {"output": content}.
    """
    if not state.messages:
        return {}

    last_message = state.messages[-1]
    if not isinstance(last_message, ToolMessage):
        return {}

    content = last_message.content
    if not content:
        return {}

    # Try to parse as JSON first
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
            else:
                # JSON array or primitive - wrap it
                return {"output": parsed}
        except json.JSONDecodeError:
            # Try to parse as Python literal (dict/list representation)
            try:
                import ast

                parsed = ast.literal_eval(content)
                if isinstance(parsed, dict):
                    return parsed
                else:
                    return {"output": parsed}
            except (ValueError, SyntaxError):
                logger.warning("Tool output is not valid JSON or Python literal")
                return {"output": content}
    elif isinstance(content, dict):
        return content
    else:
        # List or other type
        return {"output": content}


def get_message_content(msg: AnyMessage) -> str:
    if isinstance(msg, (HumanMessage, SystemMessage)):
        return msg.content if isinstance(msg.content, str) else str(msg.content)
    return str(getattr(msg, "content", "")) if hasattr(msg, "content") else ""


def _sanitize_selector_tool_names(selector):
    """Sanitize tool names in the selector's match_names for Tool scope guardrails.

    This ensures that the tool names in the selector match the sanitized tool names
    used in the actual tool nodes.

    Args:
        selector: The guardrail selector object.

    Returns:
        The selector with sanitized match_names (if applicable).
    """
    from uipath.platform.guardrails import GuardrailScope

    # Only sanitize for Tool scope guardrails
    if GuardrailScope.TOOL in selector.scopes and selector.match_names is not None:
        # Sanitize each tool name in match_names
        sanitized_names = [sanitize_tool_name(name) for name in selector.match_names]
        # Update the selector with sanitized names
        selector.match_names = sanitized_names

    return selector
