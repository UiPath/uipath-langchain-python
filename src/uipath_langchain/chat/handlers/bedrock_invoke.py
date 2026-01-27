"""Bedrock Invoke payload handler."""

from typing import Any

from langchain_core.messages import AIMessage
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.exceptions import AgentTerminationException

from .base import ModelPayloadHandler

FAULTY_STOP_REASONS: set[str] = {
    "max_tokens",
    "refusal",
    "model_context_window_exceeded",
}

STOP_REASON_MESSAGES: dict[str, tuple[str, str]] = {
    "max_tokens": (
        "Model response truncated due to max tokens limit.",
        "The model ran out of tokens while generating a response. "
        "Consider increasing max_tokens or simplifying the request.",
    ),
    "refusal": (
        "Model refused to generate response due to safety policy.",
        "The model refused to generate a response due to safety concerns. "
        "Modify your request to comply with the model's safety guidelines.",
    ),
    "model_context_window_exceeded": (
        "Context window limit exceeded.",
        "The conversation exceeded the model's context window limit. "
        "Reduce the conversation history or use a model with larger context.",
    ),
}


class BedrockInvokePayloadHandler(ModelPayloadHandler):
    """Payload handler for AWS Bedrock Invoke API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for Bedrock Invoke API."""
        return {"type": "any"}

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check Bedrock Invoke stop_reason and raise exception for faulty terminations.

        Bedrock Invoke API returns stop_reason (snake_case) in additional_kwargs.

        Args:
            response: The AIMessage response from the model

        Raises:
            AgentTerminationException: If stop_reason indicates a faulty termination
        """
        stop_reason = response.response_metadata.get("stop_reason")
        if not stop_reason:
            return

        if stop_reason in FAULTY_STOP_REASONS:
            title, detail = STOP_REASON_MESSAGES.get(
                stop_reason,
                (
                    f"Model stopped with reason: {stop_reason}",
                    f"The model terminated with stop reason '{stop_reason}'.",
                ),
            )
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title=title,
                detail=detail,
            )
