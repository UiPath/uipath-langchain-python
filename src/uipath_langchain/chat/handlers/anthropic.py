"""Anthropic Chat Completions payload handler."""

from collections.abc import Sequence
from functools import cached_property
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler, is_byo_model

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


_CACHE_CONTROL = {"type": "ephemeral"}


def _with_cache_breakpoint(message: AnyMessage) -> AnyMessage:
    content = message.content
    if isinstance(content, str):
        if not content.strip():
            return message
        blocks: list[Any] = [{"type": "text", "text": content}]
    elif content and isinstance(content[-1], dict):
        blocks = list(content)
    else:
        return message
    blocks[-1] = {**blocks[-1], "cache_control": _CACHE_CONTROL}
    return message.model_copy(update={"content": blocks})


def _with_cache_breakpoints(messages: list[AnyMessage]) -> list[AnyMessage]:
    marked = list(messages)
    targets = {len(marked) - 1}
    system = next(
        (i for i, m in enumerate(marked) if isinstance(m, SystemMessage)), None
    )
    if system is not None:
        targets.add(system)
    for i in targets:
        marked[i] = _with_cache_breakpoint(marked[i])
    return marked


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

    @cached_property
    def _caches_prompts(self) -> bool:
        return not is_byo_model(self.model)

    def apply_prompt_caching(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], dict[str, Any]]:
        if not messages or not self._caches_prompts:
            return messages, {}
        return _with_cache_breakpoints(messages), {}

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
