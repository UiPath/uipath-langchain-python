"""Bedrock Converse payload handler."""

from typing import Any

from .base import ModelPayloadHandler


class BedrockConversePayloadHandler(ModelPayloadHandler):
    """Payload handler for AWS Bedrock Converse API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for Bedrock Converse API."""
        return "any"
