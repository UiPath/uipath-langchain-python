import uuid
from langchain.messages import AIMessage, ToolCall
from langchain_core.messages.content import ContentBlock, create_tool_call


def replace_tool_calls(message: AIMessage, tool_calls: list[ToolCall]) -> AIMessage:
    """Replace tool calls in an AIMessage with new tool calls. This overwrites the entire list of tool calls.

    Args:
        message: The original AIMessage containing tool calls.
        tool_calls: The new list of ToolCall objects to replace the existing ones.

    Returns:
        A new AIMessage with the updated tool calls.
    """

    response_metadata = {
        **(message.response_metadata or {}),
        "output_version": "v1",  # we have to set this otherwise anthropic clients do not denormalize
    }

    # ToolCall from langchain.messages is not the same type as ToolCall from langchain_core.messages.content
    # they are both TypedDicts with the same fields, so it would be possible to just cast, but it's safer to map
    tool_call_blocks = [
        create_tool_call(
            name=tool_call["name"], args=tool_call["args"], id=tool_call["id"]
        )
        for tool_call in tool_calls
    ]

    content_blocks: list[ContentBlock] = [
        block for block in message.content_blocks if block["type"] != "tool_call"
    ]

    content_blocks.extend(tool_call_blocks)

    return AIMessage(
        content_blocks=content_blocks,
        tool_calls=tool_calls,
        response_metadata=response_metadata,
    )

def parse_attachment_id_from_uri(uri: str) -> str | None:
    """Parse attachment ID from a URI.

    Extracts the UUID from URIs like:
    "urn:uipath:cas:file:orchestrator:a940a416-b97b-4146-3089-08de5f4d0a87"

    Args:
        uri: The URI to parse

    Returns:
        The attachment ID if found, None otherwise
    """
    if not uri:
        return None

    # The UUID is the last segment after the final colon
    parts = uri.rsplit(":", 1)
    if len(parts) != 2:
        return None

    potential_uuid = parts[1]
    if not potential_uuid:
        return None

    # Validate it's a proper UUID and normalize to lowercase
    try:
        return str(uuid.UUID(potential_uuid))
    except (ValueError, AttributeError):
        return None
