"""OpenAI payload handler."""

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

# --- Chat Completions API constants ---

COMPLETIONS_FAULTY_REASONS: set[str] = {
    "length",
    "content_filter",
}

COMPLETIONS_ERROR_MESSAGES: dict[str, tuple[str, str]] = {
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

# --- Responses API constants ---

RESPONSES_FAULTY_REASONS: set[str] = {
    "max_output_tokens",
    "content_filter",
}

RESPONSES_ERROR_MESSAGES: dict[str, tuple[str, str]] = {
    "max_output_tokens": (
        "Model response truncated due to max tokens limit.",
        "The model ran out of tokens while generating a response. "
        "Consider increasing max_tokens or simplifying the request.",
    ),
    "content_filter": (
        "Response filtered due to content policy.",
        "The model's response was filtered due to content policy violation. "
        "Modify your request to comply with content policies.",
    ),
    "failed": (
        "Model request failed.",
        "The model request failed.",
    ),
}


class OpenAIPayloadHandler(ModelPayloadHandler):
    """Payload handler for OpenAI Chat Completions and Responses APIs.

    Automatically detects the API format from response metadata:
    - Chat Completions API uses ``finish_reason``
    - Responses API uses ``status`` / ``incomplete_details``
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
            "parallel_tool_calls": parallel_tool_calls,
            "strict": strict_mode,
        }

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check OpenAI stop reason and raise exception for faulty terminations.

        Handles both API formats:
        - Chat Completions: checks ``finish_reason`` in response_metadata
        - Responses: checks ``status`` and ``incomplete_details`` in response_metadata

        Args:
            response: The AIMessage response from the model.

        Raises:
            ChatModelError: If the stop reason indicates a faulty termination.
        """
        metadata = response.response_metadata

        # --- Responses API: status-based checks ---
        status = metadata.get("status")

        if status == "failed":
            error = metadata.get("error", {})
            error_message = error.get("message", "") if isinstance(error, dict) else ""
            title, detail = RESPONSES_ERROR_MESSAGES["failed"]
            if error_message:
                detail = f"{detail} Error: {error_message}"
            raise ChatModelError(
                code=ChatModelErrorCode.UNSUCCESSFUL_STOP_REASON,
                title=title,
                detail=detail,
                category=UiPathErrorCategory.USER,
            )

        if status == "incomplete":
            incomplete_details = metadata.get("incomplete_details", {})
            reason = (
                incomplete_details.get("reason")
                if isinstance(incomplete_details, dict)
                else None
            )
            if reason and reason in RESPONSES_FAULTY_REASONS:
                title, detail = RESPONSES_ERROR_MESSAGES.get(
                    reason,
                    (
                        f"Model response incomplete: {reason}",
                        f"The model response was incomplete due to '{reason}'.",
                    ),
                )
                raise ChatModelError(
                    code=ChatModelErrorCode.UNSUCCESSFUL_STOP_REASON,
                    title=title,
                    detail=detail,
                    category=UiPathErrorCategory.USER,
                )
            return

        # --- Chat Completions API: finish_reason-based checks ---
        finish_reason = metadata.get("finish_reason")
        if finish_reason and finish_reason in COMPLETIONS_FAULTY_REASONS:
            title, detail = COMPLETIONS_ERROR_MESSAGES.get(
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
