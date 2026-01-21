"""OpenAI payload handlers."""

from typing import Any

from .base import ModelPayloadHandler


class OpenAICompletionsPayloadHandler(ModelPayloadHandler):
    """Payload handler for OpenAI Chat Completions API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for OpenAI Completions API."""
        return "required"
