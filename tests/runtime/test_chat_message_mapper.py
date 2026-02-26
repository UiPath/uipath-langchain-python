"""Tests for UiPathChatMessagesMapper."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from uipath.core.chat import (
    UiPathConversationCitationSourceMedia,
    UiPathConversationCitationSourceUrl,
    UiPathConversationContentPart,
    UiPathConversationMessage,
    UiPathExternalValue,
    UiPathInlineValue,
)

from uipath_langchain.runtime.messages import UiPathChatMessagesMapper

# Helper timestamp string for tests
TEST_TIMESTAMP = "2025-01-15T10:30:00Z"


def create_mock_storage():
    """Create a mock storage object for testing."""
    storage = AsyncMock()
    storage.get_value = AsyncMock(return_value=None)
    storage.set_value = AsyncMock()
    return storage


class TestExtractText:
    """Tests for the _extract_text method."""

    def test_extract_text_from_string(self):
        """Should return string content as-is."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)

        result = mapper._extract_text("hello world")

        assert result == "hello world"

    def test_extract_text_from_list_with_text_parts(self):
        """Should extract text from list of content parts."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        content = [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]

        result = mapper._extract_text(content)

        assert result == "hello world"

    def test_extract_text_from_list_ignores_non_text_parts(self):
        """Should ignore non-text parts in content list."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        content = [
            {"type": "text", "text": "hello"},
            {"type": "image", "url": "http://example.com/img.png"},
            {"type": "text", "text": " world"},
        ]

        result = mapper._extract_text(content)

        assert result == "hello world"

    def test_extract_text_from_empty_list(self):
        """Should return empty string for empty list."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)

        result = mapper._extract_text([])

        assert result == ""

    def test_extract_text_from_none(self):
        """Should return empty string for None."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)

        result = mapper._extract_text(None)

        assert result == ""

    def test_extract_text_from_other_type(self):
        """Should convert other types to string."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)

        result = mapper._extract_text(123)

        assert result == "123"


class TestMapMessages:
    """Tests for the map_messages method."""

    def test_map_messages_raises_on_non_list(self):
        """Should raise TypeError when messages is not a list."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)

        with pytest.raises(TypeError, match="messages must be a list"):
            mapper.map_messages("not a list")  # type: ignore[arg-type]

    def test_map_messages_returns_empty_list_for_empty_input(self):
        """Should return empty list for empty input."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)

        result = mapper.map_messages([])

        assert result == []

    def test_map_messages_passes_through_langchain_messages(self):
        """Should return LangChain messages as-is."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]

        result = mapper.map_messages(messages)

        assert result == messages

    def test_map_messages_converts_uipath_messages(self):
        """Should convert UiPath messages to HumanMessages."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-1",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="hello world"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        assert msg.content_blocks[0]["text"] == "hello world"  # type: ignore[typeddict-item]
        assert msg.content_blocks[0]["id"] == "part-1"
        assert msg.additional_kwargs["message_id"] == "msg-1"
        assert msg.additional_kwargs["created_at"] == TEST_TIMESTAMP
        assert msg.additional_kwargs["updated_at"] == TEST_TIMESTAMP

    def test_map_messages_converts_dict_messages(self):
        """Should convert dict messages to HumanMessages."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        dict_msg = {
            "message_id": "msg-1",
            "role": "user",
            "createdAt": "2025-01-15T10:30:00Z",
            "updatedAt": "2025-01-15T10:30:00Z",
            "content_parts": [
                {
                    "content_part_id": "part-1",
                    "mime_type": "text/plain",
                    "data": {"inline": "hello from dict"},
                    "citations": [],
                    "createdAt": "2025-01-15T10:30:00Z",
                    "updatedAt": "2025-01-15T10:30:00Z",
                }
            ],
            "tool_calls": [],
            "interrupts": [],
        }

        result = mapper.map_messages([dict_msg])

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content_blocks[0]["text"] == "hello from dict"  # type: ignore[typeddict-item]

    def test_map_messages_raises_on_mixed_uipath_types(self):
        """Should raise TypeError for mixed UiPath message types."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[],
            interrupts=[],
        )

        with pytest.raises(TypeError, match="Mixed message types not supported"):
            mapper.map_messages([uipath_msg, "not a uipath message"])

    def test_map_messages_passthrough_unknown_types(self):
        """Should pass through unknown types."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        unknown = [{"unknown_field": "value"}]

        result = mapper.map_messages(unknown)

        assert result == unknown

    def test_map_messages_handles_user_message_with_multiple_content_parts(self):
        """Should create single HumanMessage with multiple content blocks."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-1",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="first part"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
                UiPathConversationContentPart(
                    content_part_id="part-2",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="second part"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        assert len(msg.content_blocks) == 2
        assert msg.content_blocks[0]["text"] == "first part"  # type: ignore[typeddict-item]
        assert msg.content_blocks[0]["id"] == "part-1"
        assert msg.content_blocks[1]["text"] == "second part"  # type: ignore[typeddict-item]
        assert msg.content_blocks[1]["id"] == "part-2"

    def test_map_messages_handles_user_message_without_content_parts(self):
        """Should handle UiPath message without content parts."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        assert msg.content == []  # Empty content_blocks list
        assert msg.additional_kwargs["message_id"] == "msg-1"

    def test_map_messages_handles_assistant_message_without_tool_calls(self):
        """Should convert assistant role to AIMessage."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-1",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="I can help with that!"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == "I can help with that!"
        assert msg.id == "msg-1"
        assert msg.additional_kwargs["message_id"] == "msg-1"

    def test_map_messages_handles_tool_calls_without_results(self):
        """Should include tool calls without results with empty content and error status."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        from uipath.core.chat import UiPathConversationToolCall

        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-1",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="Let me search for that."),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            tool_calls=[
                UiPathConversationToolCall(
                    tool_call_id="call-123",
                    name="search database",
                    input={"query": "test query"},
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        # AIMessage + ToolMessage
        assert len(result) == 2
        ai_msg = result[0]
        assert isinstance(ai_msg, AIMessage)
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["id"] == "call-123"
        assert ai_msg.tool_calls[0]["name"] == "search_database"

        tool_msg = result[1]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "call-123"
        assert tool_msg.content == ""  # Empty content for tool without result
        assert tool_msg.status == "error"  # Error status for tool without result

    def test_map_messages_includes_tool_calls_with_results(self):
        """Should create AIMessage with tool_calls AND ToolMessage for completed tool calls."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        from uipath.core.chat import (
            UiPathConversationToolCall,
            UiPathConversationToolCallResult,
        )

        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-1",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="Let me search for that."),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            tool_calls=[
                UiPathConversationToolCall(
                    tool_call_id="call-123",
                    name="search database",
                    input={"query": "test query"},
                    result=UiPathConversationToolCallResult(
                        output={"results": ["item1", "item2"]},
                        is_error=False,
                    ),
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        # Should have AIMessage + ToolMessage
        assert len(result) == 2

        # Check AIMessage
        ai_msg = result[0]
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "Let me search for that."
        assert len(ai_msg.tool_calls) == 1
        tool_call = ai_msg.tool_calls[0]
        assert (
            tool_call["name"] == "search_database"
        )  # Spaces replaced with underscores
        assert tool_call["args"] == {"query": "test query"}
        assert tool_call["id"] == "call-123"

        # Check ToolMessage
        tool_msg = result[1]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.tool_call_id == "call-123"
        assert tool_msg.content == '{"results": ["item1", "item2"]}'
        assert tool_msg.status == "success"

    def test_map_messages_includes_tool_calls_with_error_results(self):
        """Should create ToolMessage with error status for failed tool calls."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        from uipath.core.chat import (
            UiPathConversationToolCall,
            UiPathConversationToolCallResult,
        )

        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[
                UiPathConversationToolCall(
                    tool_call_id="call-456",
                    name="failing tool",
                    input={"param": "value"},
                    result=UiPathConversationToolCallResult(
                        output="Tool execution failed",
                        is_error=True,
                    ),
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 2
        tool_msg = result[1]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.status == "error"
        assert tool_msg.content == "Tool execution failed"

    def test_map_messages_includes_tool_calls_with_string_output(self):
        """Should handle string output in tool results without JSON serialization."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        from uipath.core.chat import (
            UiPathConversationToolCall,
            UiPathConversationToolCallResult,
        )

        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[
                UiPathConversationToolCall(
                    tool_call_id="call-789",
                    name="string tool",
                    input={},
                    result=UiPathConversationToolCallResult(
                        output="plain text result",
                        is_error=False,
                    ),
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 2
        tool_msg = result[1]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.content == "plain text result"

    def test_map_messages_includes_tool_calls_with_none_output(self):
        """Should handle None output in tool results as empty string."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        from uipath.core.chat import (
            UiPathConversationToolCall,
            UiPathConversationToolCallResult,
        )

        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[
                UiPathConversationToolCall(
                    tool_call_id="call-999",
                    name="none tool",
                    input={},
                    result=UiPathConversationToolCallResult(
                        output=None,
                        is_error=False,
                    ),
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 2
        tool_msg = result[1]
        assert isinstance(tool_msg, ToolMessage)
        assert tool_msg.content == ""

    def test_map_messages_includes_multiple_tool_calls_with_mixed_results(self):
        """Should handle multiple tool calls, some with results and some without."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        from uipath.core.chat import (
            UiPathConversationToolCall,
            UiPathConversationToolCallResult,
        )

        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[
                # Tool call with result
                UiPathConversationToolCall(
                    tool_call_id="call-1",
                    name="tool with result",
                    input={"a": 1},
                    result=UiPathConversationToolCallResult(
                        output={"status": "done"},
                        is_error=False,
                    ),
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
                # Tool call without result
                UiPathConversationToolCall(
                    tool_call_id="call-2",
                    name="tool without result",
                    input={"b": 2},
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        # Should have AIMessage + 2 ToolMessages (for both tool calls)
        assert len(result) == 3

        ai_msg = result[0]
        assert isinstance(ai_msg, AIMessage)
        # All tool calls are included
        assert len(ai_msg.tool_calls) == 2
        assert ai_msg.tool_calls[0]["id"] == "call-1"
        assert ai_msg.tool_calls[0]["name"] == "tool_with_result"  # Spaces replaced
        assert ai_msg.tool_calls[1]["id"] == "call-2"
        assert ai_msg.tool_calls[1]["name"] == "tool_without_result"

        # First ToolMessage (with result)
        tool_msg_1 = result[1]
        assert isinstance(tool_msg_1, ToolMessage)
        assert tool_msg_1.tool_call_id == "call-1"
        assert tool_msg_1.content == '{"status": "done"}'
        assert tool_msg_1.status == "success"

        # Second ToolMessage (without result)
        tool_msg_2 = result[2]
        assert isinstance(tool_msg_2, ToolMessage)
        assert tool_msg_2.tool_call_id == "call-2"
        assert tool_msg_2.content == ""  # Empty content for tool without result
        assert tool_msg_2.status == "error"  # Error status for tool without result

    def test_map_messages_handles_tool_calls_without_input(self):
        """Should handle tool call with None input and with result."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        from uipath.core.chat import (
            UiPathConversationToolCall,
            UiPathConversationToolCallResult,
        )

        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[
                UiPathConversationToolCall(
                    tool_call_id="call-123",
                    name="simple tool",
                    input=None,
                    result=UiPathConversationToolCallResult(
                        output="success",
                        is_error=False,
                    ),
                    timestamp=TEST_TIMESTAMP,
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 2
        msg = result[0]
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        # Should default to empty dict when input is None
        assert msg.tool_calls[0]["args"] == {}

    def test_map_messages_handles_mixed_user_and_assistant_messages(self):
        """Should handle realistic conversation with mixed message types."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        messages = [
            UiPathConversationMessage(
                message_id="msg-1",
                role="user",
                created_at=TEST_TIMESTAMP,
                updated_at=TEST_TIMESTAMP,
                content_parts=[
                    UiPathConversationContentPart(
                        content_part_id="part-1",
                        mime_type="text/plain",
                        data=UiPathInlineValue(inline="Hello"),
                        citations=[],
                        created_at=TEST_TIMESTAMP,
                        updated_at=TEST_TIMESTAMP,
                    )
                ],
                tool_calls=[],
                interrupts=[],
            ),
            UiPathConversationMessage(
                message_id="msg-2",
                role="assistant",
                created_at=TEST_TIMESTAMP,
                updated_at=TEST_TIMESTAMP,
                content_parts=[
                    UiPathConversationContentPart(
                        content_part_id="part-2",
                        mime_type="text/plain",
                        data=UiPathInlineValue(inline="Hi there!"),
                        citations=[],
                        created_at=TEST_TIMESTAMP,
                        updated_at=TEST_TIMESTAMP,
                    )
                ],
                tool_calls=[],
                interrupts=[],
            ),
            UiPathConversationMessage(
                message_id="msg-3",
                role="user",
                created_at=TEST_TIMESTAMP,
                updated_at=TEST_TIMESTAMP,
                content_parts=[
                    UiPathConversationContentPart(
                        content_part_id="part-3",
                        mime_type="text/plain",
                        data=UiPathInlineValue(inline="How are you?"),
                        citations=[],
                        created_at=TEST_TIMESTAMP,
                        updated_at=TEST_TIMESTAMP,
                    )
                ],
                tool_calls=[],
                interrupts=[],
            ),
        ]

        result = mapper.map_messages(messages)

        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], HumanMessage)

    def test_map_messages_handles_assistant_with_multiple_content_parts(self):
        """Should combine multiple content parts into single AIMessage content string."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-1",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="First part. "),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
                UiPathConversationContentPart(
                    content_part_id="part-2",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="Second part."),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        # Should create ONE AIMessage with combined content
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == "First part. Second part."

    def test_map_messages_handles_empty_content_parts_for_ai_message(self):
        """Should create valid AIMessage with no content parts."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == ""
        assert msg.additional_kwargs["message_id"] == "msg-1"
        assert msg.additional_kwargs["created_at"] == TEST_TIMESTAMP
        assert msg.additional_kwargs["updated_at"] == TEST_TIMESTAMP

    def test_map_messages_assistant_role_produces_ai_message(self):
        """Should convert assistant role messages to AIMessage."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="assistant",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-1",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="I can help with that"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                )
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, AIMessage)
        assert msg.content == "I can help with that"
        assert msg.additional_kwargs["message_id"] == "msg-1"

    def test_map_messages_external_value_produces_attachment_content(self):
        """Should include attachment metadata in content for external value content parts."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-text",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="Check this file"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
                UiPathConversationContentPart(
                    content_part_id="part-file",
                    mime_type="application/pdf",
                    data=UiPathExternalValue(
                        uri="urn:uipath:cas:file:orchestrator:a940a416-b97b-4146-3089-08de5f4d0a87"
                    ),
                    name="test.pdf",
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        # Text content block + attachment text block
        assert len(msg.content_blocks) == 2
        assert msg.content_blocks[0]["text"] == "Check this file"  # type: ignore[typeddict-item]
        assert "<uip:attachments>" in msg.content_blocks[1]["text"]  # type: ignore[typeddict-item]
        assert "a940a416-b97b-4146-3089-08de5f4d0a87" in msg.content_blocks[1]["text"]  # type: ignore[typeddict-item]
        assert "attachments" in msg.additional_kwargs
        assert msg.additional_kwargs["attachments"] == [
            {
                "id": "a940a416-b97b-4146-3089-08de5f4d0a87",
                "full_name": "test.pdf",
                "mime_type": "application/pdf",
            }
        ]

    def test_map_messages_external_value_with_empty_uri_skips_attachment(self):
        """Should skip attachment when external value has an empty URI."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-text",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="Check this file"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
                UiPathConversationContentPart(
                    content_part_id="part-file",
                    mime_type="application/pdf",
                    data=UiPathExternalValue(uri=""),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        # Only the text block, no attachment block
        assert len(msg.content_blocks) == 1
        assert msg.content_blocks[0]["text"] == "Check this file"  # type: ignore[typeddict-item]
        assert "attachments" not in msg.additional_kwargs

    def test_map_messages_external_value_with_invalid_uri_skips_attachment(self):
        """Should skip attachment when URI has no valid UUID."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-text",
                    mime_type="text/plain",
                    data=UiPathInlineValue(inline="Check this file"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
                UiPathConversationContentPart(
                    content_part_id="part-file",
                    mime_type="application/pdf",
                    data=UiPathExternalValue(uri="urn:uipath:cas:file:orchestrator:"),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        # Only the text block, no attachment block
        assert len(msg.content_blocks) == 1
        assert msg.content_blocks[0]["text"] == "Check this file"  # type: ignore[typeddict-item]
        assert "attachments" not in msg.additional_kwargs

    def test_map_messages_external_value_without_name_skips_attachment(self):
        """Should skip attachment when external value has no name."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-file",
                    mime_type="application/pdf",
                    data=UiPathExternalValue(
                        uri="urn:uipath:cas:file:orchestrator:a940a416-b97b-4146-3089-08de5f4d0a87"
                    ),
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        # No content blocks at all (external value without name is skipped)
        assert len(msg.content_blocks) == 0
        assert "attachments" not in msg.additional_kwargs

    def test_map_messages_external_value_normalizes_uppercase_uuid(self):
        """Should normalize uppercase UUID in attachment URI to lowercase."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        uipath_msg = UiPathConversationMessage(
            message_id="msg-1",
            role="user",
            created_at=TEST_TIMESTAMP,
            updated_at=TEST_TIMESTAMP,
            content_parts=[
                UiPathConversationContentPart(
                    content_part_id="part-file",
                    mime_type="application/pdf",
                    data=UiPathExternalValue(
                        uri="urn:uipath:cas:file:orchestrator:A940A416-B97B-4146-3089-08DE5F4D0A87"
                    ),
                    name="test.pdf",
                    citations=[],
                    created_at=TEST_TIMESTAMP,
                    updated_at=TEST_TIMESTAMP,
                ),
            ],
            tool_calls=[],
            interrupts=[],
        )

        result = mapper.map_messages([uipath_msg])

        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        # Only the attachment text block (no inline text part)
        assert len(msg.content_blocks) == 1
        assert "a940a416-b97b-4146-3089-08de5f4d0a87" in msg.content_blocks[0]["text"]  # type: ignore[typeddict-item]
        assert "A940A416" not in msg.content_blocks[0]["text"]  # type: ignore[typeddict-item]
        assert msg.additional_kwargs["attachments"] == [
            {
                "id": "a940a416-b97b-4146-3089-08de5f4d0a87",
                "full_name": "test.pdf",
                "mime_type": "application/pdf",
            }
        ]


class TestMapEvent:
    """Tests for the map_event method."""

    @pytest.mark.asyncio
    async def test_map_event_returns_empty_list_for_ai_chunk_without_id(self):
        """Should return empty list for AIMessageChunk without id."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        chunk = AIMessageChunk(content="hello", id=None)

        result = await mapper.map_event(chunk)

        assert result == []

    @pytest.mark.asyncio
    async def test_map_event_starts_new_message_for_new_id(self):
        """Should emit start event for new message id."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        chunk = AIMessageChunk(content="", id="msg-123")

        result = await mapper.map_event(chunk)

        assert result is not None
        assert len(result) == 1
        event = result[0]
        assert event.message_id == "msg-123"
        assert event.start is not None
        assert event.start.role == "assistant"
        assert event.content_part is not None
        assert event.content_part.start is not None

    @pytest.mark.asyncio
    async def test_map_event_tracks_seen_message_ids(self):
        """Should track seen message ids."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        chunk = AIMessageChunk(content="", id="msg-123")

        await mapper.map_event(chunk)

        assert "msg-123" in mapper.seen_message_ids

    @pytest.mark.asyncio
    async def test_map_event_emits_text_chunk_for_subsequent_messages(self):
        """Should emit text chunk event for subsequent messages with same id."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        # First chunk starts the message
        first_chunk = AIMessageChunk(content="", id="msg-123")
        await mapper.map_event(first_chunk)

        # Second chunk with text content
        second_chunk = AIMessageChunk(
            content="",
            id="msg-123",
            content_blocks=[{"type": "text", "text": "hello"}],
        )
        result = await mapper.map_event(second_chunk)

        assert result is not None
        assert len(result) == 1
        event = result[0]
        assert event.content_part is not None
        assert event.content_part.chunk is not None
        assert event.content_part.chunk.data == "hello"

    @pytest.mark.asyncio
    async def test_map_event_handles_raw_string_content(self):
        """Should handle raw string content on chunk."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        # First chunk starts the message
        first_chunk = AIMessageChunk(content="", id="msg-123")
        await mapper.map_event(first_chunk)

        # Second chunk with string content
        second_chunk = AIMessageChunk(content="raw content", id="msg-123")
        result = await mapper.map_event(second_chunk)

        assert result is not None
        assert len(result) == 1
        event = result[0]
        assert event.content_part is not None
        assert event.content_part.chunk is not None
        assert event.content_part.chunk.data == "raw content"

    @pytest.mark.asyncio
    async def test_map_event_tracks_tool_call_to_message_mapping_in_storage(self):
        """Should track tool_call_id to ai_message_id mapping in storage."""
        storage = create_mock_storage()
        storage.get_value.return_value = {}
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        # First chunk starts the message with tool_calls
        first_chunk = AIMessageChunk(
            content="",
            id="msg-123",
            tool_calls=[{"id": "tool-1", "name": "test_tool", "args": {}}],
        )
        await mapper.map_event(first_chunk)

        # Last chunk triggers tool call start events
        last_chunk = AIMessageChunk(
            content="",
            id="msg-123",
        )
        object.__setattr__(last_chunk, "chunk_position", "last")
        await mapper.map_event(last_chunk)

        # Verify storage was called with the tool call mapping
        storage.set_value.assert_called()
        call_args = storage.set_value.call_args
        assert call_args[0][0] == "test-runtime"
        assert call_args[0][1] == "chat-event-mapper"
        assert call_args[0][2] == "tool_call_map"
        assert "tool-1" in call_args[0][3]
        assert call_args[0][3]["tool-1"] == "msg-123"

    @pytest.mark.asyncio
    async def test_map_event_emits_end_event_for_last_chunk_without_tool_calls(self):
        """Should emit end event for chunk with position 'last' when no tool calls."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        # First chunk starts the message
        first_chunk = AIMessageChunk(content="", id="msg-123")
        await mapper.map_event(first_chunk)

        # Last chunk with no tool calls
        last_chunk = AIMessageChunk(content="", id="msg-123")
        object.__setattr__(last_chunk, "chunk_position", "last")

        result = await mapper.map_event(last_chunk)

        assert result is not None
        # Should have the end event
        end_event = result[-1]
        assert end_event.end is not None
        assert end_event.content_part is not None
        assert end_event.content_part.end is not None

    @pytest.mark.asyncio
    async def test_map_event_emits_tool_call_start_events_on_last_chunk(self):
        """Should emit tool call start events when chunk is last."""
        storage = create_mock_storage()
        storage.get_value.return_value = {}
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        # First chunk starts the message with tool_calls
        first_chunk = AIMessageChunk(
            content="",
            id="msg-123",
            tool_calls=[
                {"id": "tool-1", "name": "test_tool", "args": {"arg": "value"}}
            ],
        )
        await mapper.map_event(first_chunk)

        # Last chunk triggers tool call start events
        last_chunk = AIMessageChunk(
            content="",
            id="msg-123",
        )
        object.__setattr__(last_chunk, "chunk_position", "last")

        result = await mapper.map_event(last_chunk)

        assert result is not None
        # Find the tool call events - may have duplicates from accumulation
        tool_events = [e for e in result if e.tool_call is not None]
        assert len(tool_events) >= 1
        # Check that at least one has the expected tool call start
        tool_start_events = [
            e for e in tool_events if e.tool_call and e.tool_call.start is not None
        ]
        assert len(tool_start_events) >= 1
        tool_event = tool_start_events[0]
        assert tool_event.tool_call is not None
        assert tool_event.tool_call.tool_call_id == "tool-1"
        assert tool_event.tool_call.start is not None
        assert tool_event.tool_call.start.tool_name == "test_tool"

    @pytest.mark.asyncio
    async def test_map_event_handles_tool_message(self):
        """Should convert ToolMessage to tool call end event."""
        storage = create_mock_storage()
        # Pre-populate the tool call mapping in storage
        storage.get_value.return_value = {"tool-1": "msg-123"}
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        tool_msg = ToolMessage(
            content='{"result": "success"}',
            tool_call_id="tool-1",
        )

        result = await mapper.map_event(tool_msg)

        assert result is not None
        assert len(result) == 2  # tool call end event + message end event
        event = result[0]
        assert event.message_id == "msg-123"
        assert event.tool_call is not None
        assert event.tool_call.tool_call_id == "tool-1"
        assert event.tool_call.end is not None
        assert event.tool_call.end.is_error == False
        assert event.tool_call.end.output == {"result": "success"}

    @pytest.mark.asyncio
    async def test_map_event_handles_tool_message_with_error(self):
        """Should convert ToolMessage with error status to appropriate tool call end event."""
        storage = create_mock_storage()
        # Pre-populate the tool call mapping in storage
        storage.get_value.return_value = {"tool-1": "msg-123"}
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        tool_msg = ToolMessage(
            content='{"exception": "Tool execution failed"}',
            tool_call_id="tool-1",
            status="error",
        )

        result = await mapper.map_event(tool_msg)

        assert result is not None
        assert len(result) == 2  # tool call end event + message end event
        event = result[0]
        assert event.message_id == "msg-123"
        assert event.tool_call is not None
        assert event.tool_call.tool_call_id == "tool-1"
        assert event.tool_call.end is not None
        assert event.tool_call.end.is_error == True
        assert event.tool_call.end.output == {"exception": "Tool execution failed"}

    @pytest.mark.asyncio
    async def test_map_event_cleans_up_tool_mapping_after_use(self):
        """Should remove tool_call_id from storage mapping after processing ToolMessage."""
        storage = create_mock_storage()
        storage.get_value.return_value = {"tool-1": "msg-123"}
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        tool_msg = ToolMessage(content="result", tool_call_id="tool-1")
        await mapper.map_event(tool_msg)

        # Verify storage was updated without the tool-1 key
        storage.set_value.assert_called()
        call_args = storage.set_value.call_args
        assert "tool-1" not in call_args[0][3]

    @pytest.mark.asyncio
    async def test_map_event_handles_tool_message_without_mapping(self):
        """Should handle ToolMessage when no mapping exists."""
        storage = create_mock_storage()
        storage.get_value.return_value = {}  # Empty mapping
        mapper = UiPathChatMessagesMapper("test-runtime", storage)
        tool_msg = ToolMessage(content="result", tool_call_id="unknown-tool")

        with patch("uipath_langchain.runtime.messages.logger") as mock_logger:
            result = await mapper.map_event(tool_msg)

            mock_logger.error.assert_called_once()
            assert result == []

    @pytest.mark.asyncio
    async def test_map_event_handles_tool_message_without_storage(self):
        """Should handle ToolMessage when no storage is configured."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        tool_msg = ToolMessage(content="result", tool_call_id="tool-1")

        with patch("uipath_langchain.runtime.messages.logger") as mock_logger:
            result = await mapper.map_event(tool_msg)

            mock_logger.error.assert_called_once()
            assert result == []

    @pytest.mark.asyncio
    async def test_map_event_parses_json_tool_content(self):
        """Should parse JSON content in ToolMessage."""
        storage = create_mock_storage()
        storage.get_value.return_value = {"tool-1": "msg-123"}
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        tool_msg = ToolMessage(
            content='{"key": "value", "number": 42}',
            tool_call_id="tool-1",
        )

        result = await mapper.map_event(tool_msg)

        assert result is not None
        event = result[0]
        assert event.tool_call is not None
        assert event.tool_call.end is not None
        assert event.tool_call.end.output == {"key": "value", "number": 42}

    @pytest.mark.asyncio
    async def test_map_event_keeps_string_content_when_not_json(self):
        """Should keep string content when not valid JSON."""
        storage = create_mock_storage()
        storage.get_value.return_value = {"tool-1": "msg-123"}
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        tool_msg = ToolMessage(
            content="not json content",
            tool_call_id="tool-1",
        )

        result = await mapper.map_event(tool_msg)

        assert result is not None
        event = result[0]
        assert event.tool_call is not None
        assert event.tool_call.end is not None
        assert event.tool_call.end.output == "not json content"

    @pytest.mark.asyncio
    async def test_map_event_returns_none_for_system_message(self):
        """Should return None for SystemMessage."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        msg = SystemMessage(content="system prompt")

        result = await mapper.map_event(msg)

        assert result is None

    @pytest.mark.asyncio
    async def test_map_event_returns_none_for_human_message(self):
        """Should return None for HumanMessage."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        msg = HumanMessage(content="user input")

        result = await mapper.map_event(msg)

        assert result is None

    @pytest.mark.asyncio
    async def test_map_event_returns_empty_list_when_no_content_to_emit(self):
        """Should return empty list when chunk has no content to emit."""
        mapper = UiPathChatMessagesMapper("test-runtime", None)
        # First chunk starts the message
        first_chunk = AIMessageChunk(content="", id="msg-123")
        await mapper.map_event(first_chunk)

        # Empty subsequent chunk with no content blocks and no string content
        empty_chunk = AIMessageChunk(content="", id="msg-123")

        result = await mapper.map_event(empty_chunk)

        assert result == []

    @pytest.mark.asyncio
    async def test_map_event_emits_message_end_after_last_tool_result(self):
        """Should emit message end event after the last tool result for a message."""
        storage = create_mock_storage()
        # Two tool calls for the same message
        storage.get_value.side_effect = [
            {"tool-1": "msg-123", "tool-2": "msg-123"},  # First lookup
            {"tool-2": "msg-123"},  # After removing tool-1
        ]
        mapper = UiPathChatMessagesMapper("test-runtime", storage)

        # First tool result - not the last one
        tool_msg1 = ToolMessage(content="result1", tool_call_id="tool-1")
        result1 = await mapper.map_event(tool_msg1)

        # Should have tool call end but NOT message end
        assert result1 is not None
        assert len(result1) == 1  # Only tool call end
        assert result1[0].tool_call is not None
        assert result1[0].end is None

        # Second tool result - the last one
        storage.get_value.side_effect = [
            {"tool-2": "msg-123"},  # Lookup for tool-2
        ]
        tool_msg2 = ToolMessage(content="result2", tool_call_id="tool-2")
        result2 = await mapper.map_event(tool_msg2)

        # Should have both tool call end AND message end
        assert result2 is not None
        assert len(result2) == 2  # Tool call end + message end
        assert result2[0].tool_call is not None
        assert result2[1].end is not None


class TestMapLangChainMessagesToUiPathMessageData:
    """Tests for map_langchain_messages_to_uipath_message_data_list static method."""

    def test_converts_empty_messages_correctly(self):
        """Should return empty list when input messages list is empty."""
        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                []
            )
        )

        assert result == []

    def test_converts_human_message_to_user_role(self):
        """Should convert HumanMessage to user role message."""
        messages: list[AnyMessage] = [HumanMessage(content="Hello")]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages
            )
        )

        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].content_parts) == 1
        assert result[0].content_parts[0].mime_type == "text/plain"
        assert isinstance(result[0].content_parts[0].data, UiPathInlineValue)
        assert result[0].content_parts[0].data.inline == "Hello"

    def test_converts_ai_message_to_assistant_role(self):
        """Should convert AIMessage to assistant role message."""
        messages: list[AnyMessage] = [AIMessage(content="Hi there")]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages
            )
        )

        assert len(result) == 1
        assert result[0].role == "assistant"
        assert len(result[0].content_parts) == 1
        assert result[0].content_parts[0].mime_type == "text/markdown"
        assert isinstance(result[0].content_parts[0].data, UiPathInlineValue)
        assert result[0].content_parts[0].data.inline == "Hi there"

    def test_converts_ai_message_with_tool_calls(self):
        """Should include tool calls in converted AI message."""
        messages: list[AnyMessage] = [
            AIMessage(
                content="Let me search",
                tool_calls=[
                    {"name": "search", "args": {"query": "test"}, "id": "call1"}
                ],
            )
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages, include_tool_results=False
            )
        )

        assert len(result) == 1
        assert result[0].role == "assistant"
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].name == "search"
        assert result[0].tool_calls[0].input == {"query": "test"}

    def test_includes_tool_results_when_enabled(self):
        """Should include tool results in tool calls when include_tool_results=True."""
        messages: list[AnyMessage] = [
            AIMessage(
                content="Using tool",
                tool_calls=[{"name": "test_tool", "args": {}, "id": "call1"}],
            ),
            ToolMessage(
                content='{"status": "success"}', tool_call_id="call1", status="success"
            ),
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages, include_tool_results=True
            )
        )

        assert len(result) == 1  # Only AI message, tool message merged in
        assert result[0].role == "assistant"
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0].result is not None
        assert result[0].tool_calls[0].result.output == {"status": "success"}
        assert result[0].tool_calls[0].result.is_error is False

    def test_excludes_tool_results_when_disabled(self):
        """Should exclude tool results when include_tool_results=False."""
        messages: list[AnyMessage] = [
            AIMessage(
                content="Using tool",
                tool_calls=[{"name": "test_tool", "args": {}, "id": "call1"}],
            ),
            ToolMessage(
                content='{"status": "success"}', tool_call_id="call1", status="success"
            ),
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages, include_tool_results=False
            )
        )

        assert len(result) == 1
        assert result[0].role == "assistant"
        assert len(result[0].tool_calls) == 1
        # Tool call should not have result when include_tool_results=False
        assert result[0].tool_calls[0].result is None

    def test_handles_tool_error_status(self):
        """Should mark tool result as error when status is error."""
        messages: list[AnyMessage] = [
            AIMessage(
                content="Trying tool",
                tool_calls=[{"name": "failing_tool", "args": {}, "id": "call1"}],
            ),
            ToolMessage(content="Error occurred", tool_call_id="call1", status="error"),
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages, include_tool_results=True
            )
        )

        assert len(result) == 1
        assert result[0].tool_calls[0].result is not None
        assert result[0].tool_calls[0].result.is_error is True
        assert result[0].tool_calls[0].result.output == "Error occurred"

    def test_parses_json_tool_results(self):
        """Should parse JSON string results back to dict."""
        messages: list[AnyMessage] = [
            AIMessage(
                content="Using tool",
                tool_calls=[{"name": "test_tool", "args": {}, "id": "call1"}],
            ),
            ToolMessage(
                content='{"data": [1, 2, 3], "count": 3}', tool_call_id="call1"
            ),
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages, include_tool_results=True
            )
        )

        assert result[0].tool_calls[0].result is not None
        assert result[0].tool_calls[0].result.output == {"data": [1, 2, 3], "count": 3}

    def test_keeps_non_json_tool_results_as_string(self):
        """Should keep non-JSON results as strings."""
        messages: list[AnyMessage] = [
            AIMessage(
                content="Using tool",
                tool_calls=[{"name": "test_tool", "args": {}, "id": "call1"}],
            ),
            ToolMessage(content="plain text result", tool_call_id="call1"),
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages, include_tool_results=True
            )
        )

        assert result[0].tool_calls[0].result is not None
        assert result[0].tool_calls[0].result.output == "plain text result"

    def test_handles_mixed_message_types(self):
        """Should handle conversation with mixed message types including tools."""
        messages: list[AnyMessage] = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="Search for data"),
            AIMessage(
                content="Let me search",
                tool_calls=[
                    {"name": "search_tool", "args": {"query": "data"}, "id": "call1"}
                ],
            ),
            ToolMessage(
                content='{"results": ["item1", "item2"]}', tool_call_id="call1"
            ),
            AIMessage(content="I found the data"),
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages, include_tool_results=True
            )
        )

        # Should skip ToolMessages, only convert Human and AI messages
        assert len(result) == 5
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[2].role == "user"
        assert result[3].role == "assistant"
        assert len(result[3].tool_calls) == 1
        assert result[3].tool_calls[0].result is not None
        assert result[4].role == "assistant"

    def test_handles_empty_message_list(self):
        """Should return empty list for empty input."""
        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                []
            )
        )

        assert result == []

    def test_handles_empty_content_messages(self):
        """Should handle messages with empty content."""
        messages: list[AnyMessage] = [
            HumanMessage(content=""),
            AIMessage(content=""),
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages
            )
        )

        assert len(result) == 2
        # Empty content should result in no text content-parts
        assert len(result[0].content_parts) == 0
        assert len(result[1].content_parts) == 0

    def test_extracts_text_from_content_blocks(self):
        """Should extract text from complex content block structures."""
        messages: list[AnyMessage] = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "first part"},
                    {"type": "text", "text": " second part"},
                ]
            )
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages
            )
        )

        assert len(result) == 1
        assert len(result[0].content_parts) == 1
        assert isinstance(result[0].content_parts[0].data, UiPathInlineValue)
        assert result[0].content_parts[0].data.inline == "first part second part"


class TestMapLangChainAIMessageCitations:
    """Tests for citation extraction in _map_langchain_ai_message_to_uipath_message_data."""

    def test_ai_message_with_citation_tags_populates_citations(self):
        """AIMessage with inline citation tags should have citations populated and text cleaned."""
        messages: list[AnyMessage] = [
            AIMessage(
                content='Some fact<uip:cite title="Doc" url="https://doc.com" /> and more.'
            )
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages
            )
        )

        assert len(result) == 1
        part = result[0].content_parts[0]
        assert isinstance(part.data, UiPathInlineValue)
        assert part.data.inline == "Some fact and more."
        assert len(part.citations) == 1
        assert part.citations[0].offset == 0
        assert part.citations[0].length == 9  # "Some fact"
        source = part.citations[0].sources[0]
        assert isinstance(source, UiPathConversationCitationSourceUrl)
        assert source.url == "https://doc.com"
        assert source.title == "Doc"

    def test_ai_message_without_citation_tags_has_empty_citations(self):
        """AIMessage without citation tags should have empty citations list."""
        messages: list[AnyMessage] = [AIMessage(content="Plain text response")]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages
            )
        )

        assert len(result) == 1
        part = result[0].content_parts[0]
        assert isinstance(part.data, UiPathInlineValue)
        assert part.data.inline == "Plain text response"
        assert part.citations == []

    def test_ai_message_with_media_citation(self):
        """AIMessage with reference/media citation tag should produce media source."""
        messages: list[AnyMessage] = [
            AIMessage(
                content='A finding<uip:cite title="Report.pdf" reference="https://r.com" page_number="3" />'
            )
        ]

        result = (
            UiPathChatMessagesMapper.map_langchain_messages_to_uipath_message_data_list(
                messages
            )
        )

        assert len(result) == 1
        part = result[0].content_parts[0]
        assert isinstance(part.data, UiPathInlineValue)
        assert part.data.inline == "A finding"
        assert len(part.citations) == 1
        source = part.citations[0].sources[0]
        assert isinstance(source, UiPathConversationCitationSourceMedia)
        assert source.download_url == "https://r.com"
        assert source.page_number == "3"
