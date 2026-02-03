"""Helper functions for chat messages manipulation."""

import uuid
from typing import Any, cast

from langchain_core.messages import BaseMessage, ContentBlock
from opentelemetry import trace


def append_content_blocks_to_message(
    message: BaseMessage,
    content_blocks: list[ContentBlock],
) -> BaseMessage:
    """Append content blocks to a message.

    Args:
        message: The original message (any BaseMessage subclass)
        content_blocks: Content blocks to append

    Returns:
        New message of the same type with appended content blocks
    """
    if not content_blocks:
        return message

    existing_content_blocks = list(message.content_blocks)
    existing_content_blocks.extend(content_blocks)

    return type(message)(content_blocks=existing_content_blocks)


def extract_text_content(message: BaseMessage) -> str:
    """Extract text content from an AI message.

    Extracts text from content blocks using duck typing to support both
    dict-like objects (with 'text' key) and objects with 'text' attribute.

    Args:
        message: The AI message to extract text from

    Returns:
        Extracted text content, with multiple text parts joined by newlines
    """
    content_blocks = message.content_blocks

    text_parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, str):
            text_parts.append(block)
        else:
            text: str | None = None
            if isinstance(block, dict):
                text_value = cast(dict[str, Any], block).get("text", "")
                if isinstance(text_value, str):
                    text = text_value
            elif hasattr(block, "text"):
                text_attr = getattr(block, "text", "")
                if isinstance(text_attr, str):
                    text = text_attr

            if text:
                text_parts.append(text)

    return "\n".join(text_parts) if text_parts else ""


def get_action_id() -> str | None:
    """Get the current trace ID as a UUID for the X-UiPath-LlmGateway-ActionId header.

    Returns:
        The trace ID formatted as a UUID string, or None if no trace is active.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.trace_id:
        return str(uuid.UUID(int=ctx.trace_id))
    return None
