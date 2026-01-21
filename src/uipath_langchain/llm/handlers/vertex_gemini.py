"""Vertex Gemini payload handler."""

from typing import Any

from .base import ModelPayloadHandler


class VertexGeminiPayloadHandler(ModelPayloadHandler):
    """Payload handler for Google Vertex AI Gemini API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for Vertex Gemini API."""
        return "any"
