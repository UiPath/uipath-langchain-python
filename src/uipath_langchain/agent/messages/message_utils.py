from langchain.messages import AIMessage, ToolCall


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

    content_blocks = [
        block for block in message.content_blocks if block["type"] != "tool_call"
    ]

    content_blocks.extend(tool_calls)

    return AIMessage(
        content_blocks=content_blocks,
        tool_calls=tool_calls,
        response_metadata=response_metadata,
    )
