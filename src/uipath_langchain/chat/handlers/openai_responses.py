"""OpenAI payload handlers."""

from langchain_core.messages import AIMessage
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

FAULTY_INCOMPLETE_REASONS: set[str] = {
    "max_output_tokens",
    "content_filter",
}

STOP_REASON_MESSAGES: dict[str, tuple[str, str]] = {
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


class OpenAIResponsesPayloadHandler(ModelPayloadHandler):
    """Payload handler for OpenAI Responses API."""

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check OpenAI Responses API status and raise exception for faulty terminations.

        OpenAI Responses API returns status and incomplete_details in response_metadata.
        - status == "failed": Always an error
        - status == "incomplete": Check incomplete_details.reason

        Args:
            response: The AIMessage response from the model

        Raises:
            ChatModelError: If status indicates a faulty termination
        """
        status = response.response_metadata.get("status")

        if status == "failed":
            error = response.response_metadata.get("error", {})
            error_message = error.get("message", "") if isinstance(error, dict) else ""
            title, detail = STOP_REASON_MESSAGES["failed"]
            if error_message:
                detail = f"{detail} Error: {error_message}"
            raise ChatModelError(
                code=ChatModelErrorCode.UNSUCCESSFUL_STOP_REASON,
                title=title,
                detail=detail,
                category=UiPathErrorCategory.USER,
            )

        if status == "incomplete":
            incomplete_details = response.response_metadata.get(
                "incomplete_details", {}
            )
            reason = (
                incomplete_details.get("reason")
                if isinstance(incomplete_details, dict)
                else None
            )
            if reason and reason in FAULTY_INCOMPLETE_REASONS:
                title, detail = STOP_REASON_MESSAGES.get(
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
