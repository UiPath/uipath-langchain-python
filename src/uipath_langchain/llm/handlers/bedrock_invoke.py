"""Bedrock Invoke payload handler."""

from typing import Any

from .base import ModelPayloadHandler


class BedrockInvokePayloadHandler(ModelPayloadHandler):
    """Payload handler for AWS Bedrock Invoke API."""

    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get tool_choice value for Bedrock Invoke API."""
        return {"type": "any"}
