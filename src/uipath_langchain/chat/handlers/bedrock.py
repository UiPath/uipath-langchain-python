"""AWS Bedrock payload handler."""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

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


class BedrockPayloadHandler(ModelPayloadHandler):
    """Payload handler for AWS Bedrock Converse and Invoke APIs.

    Automatically detects the API format from response metadata:
    - Converse API uses ``stopReason`` (camelCase)
    - Invoke API uses ``stop_reason`` (snake_case)
    """

    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool = True,
        strict_mode: bool = False,
    ) -> dict[str, Any]:
        return {
            "tool_choice": tool_choice,
        }

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check Bedrock stop reason and raise exception for faulty terminations.

        Handles both API formats:
        - Converse: checks ``stopReason`` (camelCase) in response_metadata
        - Invoke: checks ``stop_reason`` (snake_case) in response_metadata

        Args:
            response: The AIMessage response from the model.

        Raises:
            ChatModelError: If the stop reason indicates a faulty termination.
        """
        metadata = response.response_metadata

        # --- Converse API: stopReason (camelCase) ---
        stop_reason = metadata.get("stopReason")
        if stop_reason:
            if stop_reason in CONVERSE_FAULTY_REASONS:
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
            return

        # --- Invoke API: stop_reason (snake_case) ---
        stop_reason = metadata.get("stop_reason")
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
