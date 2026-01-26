"""Tests for simulated tool call logging and tracing."""

import pytest
from langchain_core.messages import AIMessageChunk, ToolCallChunk
from uipath_langchain.runtime.messages import UiPathChatMessagesMapper


def test_simulated_tool_call_metadata_added():
    """Test that simulated flag from response_metadata is added to tool call event metadata."""
    mapper = UiPathChatMessagesMapper()

    # Create first chunk with tool call
    chunk1 = AIMessageChunk(
        content="",
        id="test-message-id",
        response_metadata={"simulated": True},
        content_blocks=[
            ToolCallChunk(
                type="tool_call_chunk",
                id="call_123",
                name="test_tool",
                args={"arg1": "value1"},
            )
        ],
    )

    # Process first chunk
    events1 = mapper.map_event(chunk1)
    assert events1 is not None

    # Create last chunk
    chunk2 = AIMessageChunk(
        content="",
        id="test-message-id",
        chunk_position="last",
    )

    # Process last chunk - this should emit tool call start event
    events2 = mapper.map_event(chunk2)
    assert events2 is not None
    assert len(events2) > 0

    # Find the tool call start event
    tool_call_event = None
    for event in events2:
        if event.tool_call and event.tool_call.start:
            tool_call_event = event
            break

    assert tool_call_event is not None, "Tool call start event should be emitted"
    assert tool_call_event.tool_call.start.metadata is not None
    assert tool_call_event.tool_call.start.metadata.get("simulated") is True


def test_non_simulated_tool_call_no_metadata():
    """Test that tool calls without simulated flag don't have metadata set."""
    mapper = UiPathChatMessagesMapper()

    # Create first chunk with tool call (no simulated flag)
    chunk1 = AIMessageChunk(
        content="",
        id="test-message-id-2",
        content_blocks=[
            ToolCallChunk(
                type="tool_call_chunk",
                id="call_456",
                name="test_tool",
                args={"arg1": "value1"},
            )
        ],
    )

    # Process first chunk
    events1 = mapper.map_event(chunk1)
    assert events1 is not None

    # Create last chunk
    chunk2 = AIMessageChunk(
        content="",
        id="test-message-id-2",
        chunk_position="last",
    )

    # Process last chunk
    events2 = mapper.map_event(chunk2)
    assert events2 is not None
    assert len(events2) > 0

    # Find the tool call start event
    tool_call_event = None
    for event in events2:
        if event.tool_call and event.tool_call.start:
            tool_call_event = event
            break

    assert tool_call_event is not None
    # Metadata should be None or empty when simulated flag is not present
    assert (
        tool_call_event.tool_call.start.metadata is None
        or tool_call_event.tool_call.start.metadata == {}
    )


def test_simulated_false_not_added():
    """Test that simulated=False is still added to metadata."""
    mapper = UiPathChatMessagesMapper()

    # Create first chunk with tool call and simulated=False
    chunk1 = AIMessageChunk(
        content="",
        id="test-message-id-3",
        response_metadata={"simulated": False},
        content_blocks=[
            ToolCallChunk(
                type="tool_call_chunk",
                id="call_789",
                name="test_tool",
                args={"arg1": "value1"},
            )
        ],
    )

    # Process first chunk
    events1 = mapper.map_event(chunk1)
    assert events1 is not None

    # Create last chunk
    chunk2 = AIMessageChunk(
        content="",
        id="test-message-id-3",
        chunk_position="last",
    )

    # Process last chunk
    events2 = mapper.map_event(chunk2)
    assert events2 is not None
    assert len(events2) > 0

    # Find the tool call start event
    tool_call_event = None
    for event in events2:
        if event.tool_call and event.tool_call.start:
            tool_call_event = event
            break

    assert tool_call_event is not None
    assert tool_call_event.tool_call.start.metadata is not None
    assert tool_call_event.tool_call.start.metadata.get("simulated") is False
