"""OpenAI Chat Completions payload handler."""

from typing import Any

from langchain_core.messages import AIMessage
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

FAULTY_FINISH_REASONS: set[str] = {
    "length",
    "content_filter",
}

FINISH_REASON_MESSAGES: dict[str, tuple[str, str]] = {
    "length": (
        "Model response truncated due to max tokens limit.",
        "The model ran out of tokens while generating a response. "
        "Consider increasing max_tokens or simplifying the request.",
    ),
    "content_filter": (
        "Response filtered due to content policy.",
        "The model's response was filtered due to content policy violation. "
        "Modify your request to comply with content policies.",
    ),
}


class OpenAICompletionsPayloadHandler(ModelPayloadHandler):
    """Payload handler for OpenAI Chat Completions API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for OpenAI Completions API."""
        return "required"

    def get_parallel_tool_calls_kwargs(
        self, parallel_tool_calls: bool
    ) -> dict[str, Any]:
        return {"parallel_tool_calls": parallel_tool_calls}

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check OpenAI finish_reason and raise exception for faulty terminations.

        OpenAI Chat Completions API returns finish_reason in response_metadata.

        Args:
            response: The AIMessage response from the model

        Raises:
            ChatModelError: If finish_reason indicates a faulty termination
        """
        finish_reason = response.response_metadata.get("finish_reason")
        if not finish_reason:
            return

        if finish_reason in FAULTY_FINISH_REASONS:
            title, detail = FINISH_REASON_MESSAGES.get(
                finish_reason,
                (
                    f"Model stopped with reason: {finish_reason}",
                    f"The model terminated with finish reason '{finish_reason}'.",
                ),
            )
            raise ChatModelError(
                code=ChatModelErrorCode.UNSUCCESSFUL_STOP_REASON,
                title=title,
                detail=detail,
                category=UiPathErrorCategory.USER,
            )
