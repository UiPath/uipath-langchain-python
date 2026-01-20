"""OpenAI payload handlers."""

from typing import Any

from .base import ModelPayloadHandler


class OpenAIResponsesPayloadHandler(ModelPayloadHandler):
    """Payload handler for OpenAI Responses API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for OpenAI Responses API."""
        return "required"
