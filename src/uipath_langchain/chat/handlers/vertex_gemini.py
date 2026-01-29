"""Vertex Gemini payload handler."""

from langchain_core.messages import AIMessage
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import ChatModelError, ChatModelErrorCode
from .base import ModelPayloadHandler

FAULTY_FINISH_REASONS: set[str] = {
    "FINISH_REASON_UNSPECIFIED",
    "MAX_TOKENS",
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "MALFORMED_FUNCTION_CALL",
    "OTHER",
    "MODEL_ARMOR",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "IMAGE_OTHER",
    "UNEXPECTED_TOOL_CALL",
    "NO_IMAGE",
}

FINISH_REASON_MESSAGES: dict[str, tuple[str, str]] = {
    "FINISH_REASON_UNSPECIFIED": (
        "Model stopped for an unspecified reason.",
        "The model terminated without providing a specific finish reason. "
        "This may indicate an unexpected issue. Try again or contact support.",
    ),
    "MAX_TOKENS": (
        "Model response truncated due to max tokens limit.",
        "The model ran out of tokens while generating a response. "
        "Consider increasing max_tokens or simplifying the request.",
    ),
    "SAFETY": (
        "Response blocked by safety filter.",
        "The model's response was blocked due to safety concerns. "
        "Modify your request to comply with the model's safety guidelines.",
    ),
    "RECITATION": (
        "Response blocked due to recitation detection.",
        "The model's response was flagged for potential recitation of training data. "
        "Try rephrasing your request to generate original content.",
    ),
    "BLOCKLIST": (
        "Response blocked due to forbidden terms.",
        "The model's response contained terms from a blocklist. "
        "Modify your request to avoid restricted content.",
    ),
    "PROHIBITED_CONTENT": (
        "Response blocked due to prohibited content.",
        "The model's response was blocked for containing prohibited content. "
        "Modify your request to comply with content policies.",
    ),
    "SPII": (
        "Response blocked due to sensitive personal information.",
        "The model's response was blocked for containing sensitive personally "
        "identifiable information (SPII). Modify your request to avoid PII.",
    ),
    "MALFORMED_FUNCTION_CALL": (
        "Model generated an invalid function call.",
        "The model attempted to make a function call but the format was invalid. "
        "This may indicate an issue with the definitions of the tools or with the model reaching max tokens.",
    ),
    "OTHER": (
        "Model stopped for an unknown reason.",
        "The model terminated for an unspecified reason. "
        "Try rephrasing your request or contact support if the issue persists.",
    ),
    "MODEL_ARMOR": (
        "Response blocked by Model Armor.",
        "The model's response was blocked by Vertex AI Model Armor policy. "
        "Review Model Armor configuration or modify your request.",
    ),
    "IMAGE_SAFETY": (
        "Response blocked due to image safety concerns.",
        "The model's response was blocked because generated images violated safety policies. "
        "Modify your request to avoid generating unsafe image content.",
    ),
    "IMAGE_PROHIBITED_CONTENT": (
        "Generated image blocked due to prohibited content.",
        "The generated image may contain prohibited content. "
        "Modify your request to avoid generating prohibited image content.",
    ),
    "IMAGE_RECITATION": (
        "Generated image blocked due to recitation detection.",
        "The generated image may be a recitation from a source. "
        "Modify your request to generate original image content.",
    ),
    "IMAGE_OTHER": (
        "Image generation stopped for an unspecified reason.",
        "The image generation stopped for a reason not otherwise specified. "
        "Try rephrasing your request or contact support if the issue persists.",
    ),
    "UNEXPECTED_TOOL_CALL": (
        "Model made an unexpected tool call.",
        "The model attempted to call a tool that was not expected or is semantically invalid. "
        "This may indicate a mismatch between tool definitions and model behavior.",
    ),
    "NO_IMAGE": (
        "Model failed to generate expected image.",
        "The model was expected to generate an image but did not produce one. "
        "Try rephrasing your request or simplifying the image generation prompt.",
    ),
}


class VertexGeminiPayloadHandler(ModelPayloadHandler):
    """Payload handler for Google Vertex AI Gemini API."""

    def check_stop_reason(self, response: AIMessage) -> None:
        """Check Vertex Gemini finishReason and raise exception for faulty terminations.

        Vertex Gemini API returns finishReason (SCREAMING_SNAKE_CASE) in response_metadata.

        Args:
            response: The AIMessage response from the model

        Raises:
            ChatModelError: If finishReason indicates a faulty termination
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
