"""AWS Bedrock payload handlers."""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

logger = logging.getLogger(__name__)


# --- Converse API constants ---

CONVERSE_FAULTY_REASONS: set[str] = {
    "max_tokens",
    "guardrail_intervened",
    "content_filtered",
    "model_context_window_exceeded",
}

CONVERSE_ERROR_MESSAGES: dict[str, tuple[str, str]] = {
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

# --- Invoke API constants ---

INVOKE_FAULTY_REASONS: set[str] = {
    "max_tokens",
    "refusal",
    "model_context_window_exceeded",
}

INVOKE_ERROR_MESSAGES: dict[str, tuple[str, str]] = {
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
    """Payload handler for ``ChatBedrock`` (AWS Bedrock Invoke API).

    - Supports ``disable_parallel_tool_use``; ``strict`` is not supported.
    - Stop reason field: ``stop_reason`` (snake_case).
    """

    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool | None = None,
        strict_mode: bool | None = None,
    ) -> dict[str, Any]:
        _thinking = (getattr(self.model, "model_kwargs", None) or {}).get("thinking")
        thinking_enabled = (
            isinstance(_thinking, dict) and _thinking.get("type") == "enabled"
        )
        # Anthropic models via Invoke API don't support forced tool use with extended thinking
        if thinking_enabled and tool_choice == "any":
            logger.warning(
                "Thinking is enabled for the model, but tool_choice is 'any'. "
                "Changing tool_choice to 'auto' to keep the same behaviour as ChatAnthropicBedrock."
            )
            tool_choice = "auto"
        kwargs: dict[str, Any] = {"tool_choice": tool_choice}
        if parallel_tool_calls is False:
            kwargs["disable_parallel_tool_use"] = True
        return kwargs

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check ``stop_reason`` (snake_case) and raise for faulty terminations.

        Args:
            response: The AIMessage response from the model.

        Raises:
            ChatModelError: If the stop reason indicates a faulty termination.
        """
        stop_reason = response.response_metadata.get("stop_reason")
        if stop_reason and stop_reason in INVOKE_FAULTY_REASONS:
            title, detail = INVOKE_ERROR_MESSAGES.get(
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


class BedrockConversePayloadHandler(ModelPayloadHandler):
    """Payload handler for ``ChatBedrockConverse`` (AWS Bedrock Converse API).

    - Supports ``strict``; ``parallel_tool_calls`` is not supported.
    - Stop reason field: ``stopReason`` (camelCase).
    """

    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool | None = None,
        strict_mode: bool | None = None,
    ) -> dict[str, Any]:
        _thinking = (
            getattr(self.model, "additional_model_request_fields", None) or {}
        ).get("thinking")
        thinking_enabled = (
            isinstance(_thinking, dict) and _thinking.get("type") == "enabled"
        )
        # Anthropic models via Converse API don't support forced tool use with extended thinking
        if thinking_enabled and tool_choice == "any":
            logger.warning(
                "Thinking is enabled for the model, but tool_choice is 'any'. "
                "Changing tool_choice to 'auto' to keep the same behaviour as ChatAnthropicBedrock."
            )
            tool_choice = "auto"
        kwargs: dict[str, Any] = {"tool_choice": tool_choice}
        if strict_mode is True:
            kwargs["strict"] = True
        return kwargs

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check ``stopReason`` (camelCase) and raise for faulty terminations.

        Args:
            response: The AIMessage response from the model.

        Raises:
            ChatModelError: If the stop reason indicates a faulty termination.
        """
        stop_reason = response.response_metadata.get("stopReason")
        if stop_reason and stop_reason in CONVERSE_FAULTY_REASONS:
            title, detail = CONVERSE_ERROR_MESSAGES.get(
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
