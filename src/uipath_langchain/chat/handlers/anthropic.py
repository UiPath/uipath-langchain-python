"""Anthropic Chat Completions payload handler."""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

FAULTY_STOP_REASONS: set[str] = {
    "max_tokens",
    "refusal",
    "model_context_window_exceeded",
}

STOP_REASON_MESSAGES: dict[str, tuple[str, str]] = {
    "max_tokens": (
        "Response truncated due to max_tokens limit.",
        "Claude stopped because it reached the max_tokens limit specified in your request. "
        "Consider increasing max_tokens or making another request to continue.",
    ),
    "refusal": (
        "Claude refused to generate a response.",
        "Claude declined to respond due to safety concerns. "
        "Consider rephrasing or modifying your request.",
    ),
    "model_context_window_exceeded": (
        "Response limited by context window.",
        "Claude stopped because it reached the model's context window limit. "
        "The response is still valid but was limited by the context window.",
    ),
}


class AnthropicPayloadHandler(ModelPayloadHandler):
    """Payload handler for Anthropic API."""

    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool = True,
        strict_mode: bool = False,
    ) -> dict[str, Any]:
        return {
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "strict": strict_mode,
        }

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check Anthropic stop_reason and raise exception for faulty terminations.

        Anthropic Chat Completions API returns stop_reason in response_metadata.

        Args:
            response: The AIMessage response from the model

        Raises:
            ChatModelError: If stop_reason indicates a faulty termination
        """
        stop_reason = response.response_metadata.get("stop_reason")
        if not stop_reason:
            return

        if stop_reason in FAULTY_STOP_REASONS:
            title, detail = STOP_REASON_MESSAGES.get(
                stop_reason,
                (
                    f"Model stopped with reason: {stop_reason}",
                    f"The model terminated with finish reason '{stop_reason}'.",
                ),
            )
            raise ChatModelError(
                code=ChatModelErrorCode.UNSUCCESSFUL_STOP_REASON,
                title=title,
                detail=detail,
                category=UiPathErrorCategory.USER,
            )
