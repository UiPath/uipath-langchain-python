"""Bedrock Converse payload handler."""

from typing import Any

from langchain_core.messages import AIMessage
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

FAULTY_STOP_REASONS: set[str] = {
    "max_tokens",
    "guardrail_intervened",
    "content_filtered",
    "model_context_window_exceeded",
}

STOP_REASON_MESSAGES: dict[str, tuple[str, str]] = {
    "max_tokens": (
        "Model response truncated due to max tokens limit.",
        "The model ran out of tokens while generating a response. "
        "Consider increasing max_tokens or simplifying the request.",
    ),
    "guardrail_intervened": (
        "Response blocked by AWS Bedrock guardrail.",
        "An AWS Bedrock guardrail policy blocked or modified the model's response. Review your request",
    ),
    "content_filtered": (
        "Response filtered due to content policy.",
        "The model's response was filtered due to content policy violation. "
        "Modify your request to comply with content policies.",
    ),
    "model_context_window_exceeded": (
        "Context window limit exceeded.",
        "The conversation exceeded the model's context window limit. "
        "Reduce the conversation history or use a model with larger context.",
    ),
}


class BedrockConversePayloadHandler(ModelPayloadHandler):
    """Payload handler for AWS Bedrock Converse API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for Bedrock Converse API."""
        return "any"

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check Bedrock Converse stopReason and raise exception for faulty terminations.

        Bedrock Converse API returns stopReason (camelCase) in response_metadata.

        Args:
            response: The AIMessage response from the model

        Raises:
            ChatModelError: If stopReason indicates a faulty termination
        """
        stop_reason = response.response_metadata.get("stopReason")
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
            raise ChatModelError(
                code=ChatModelErrorCode.UNSUCCESSFUL_STOP_REASON,
                title=title,
                detail=detail,
                category=UiPathErrorCategory.USER,
            )
