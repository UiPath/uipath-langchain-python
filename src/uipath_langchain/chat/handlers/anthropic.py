"""Anthropic Chat Completions payload handler."""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler


def anthropic_thinking_type(model: Any) -> str | None:
    """The Anthropic thinking mode for a model ("enabled"/"adaptive"/...), or None.

    Reads the `thinking` dict wherever the transport puts it: native `thinking`, Bedrock
    Invoke `model_kwargs`, Bedrock Converse `additional_model_request_fields`. Only
    Anthropic uses this shape — OpenAI/Gemini use other knobs — so non-Anthropic models
    return None. Null-safe.
    """
    invoke = getattr(model, "model_kwargs", None) or {}
    converse = getattr(model, "additional_model_request_fields", None) or {}
    candidates = (
        getattr(model, "thinking", None),
        invoke.get("thinking") if isinstance(invoke, dict) else None,
        converse.get("thinking") if isinstance(converse, dict) else None,
    )
    for thinking in candidates:
        if isinstance(thinking, dict) and isinstance(thinking.get("type"), str):
            return thinking["type"]
    return None


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
        parallel_tool_calls: bool | None = None,
        strict_mode: bool | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"tool_choice": tool_choice}
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        if strict_mode is True:
            kwargs["strict"] = True
        return kwargs

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
