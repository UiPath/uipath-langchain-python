"""Tests for agent/messages/message_utils.py module."""

from langchain.messages import AIMessage, ToolCall
from langchain_core.messages.content import (
    ContentBlock,
    create_text_block,
    create_tool_call,
)

from uipath_langchain.agent.messages.message_utils import (
    parse_attachment_id_from_uri,
    replace_tool_calls,
)


class TestReplaceToolCalls:
    """Test cases for replace_tool_calls function."""

    def test_replace_tool_calls_basic(self):
        """Test basic tool call replacement."""
        original_tool_calls = [
            ToolCall(name="old_tool", args={"param": "value"}, id="old_id")
        ]
        original_content_blocks: list[ContentBlock] = [
            create_text_block("Test message"),
            create_tool_call(name="old_tool", args={"param": "value"}, id="old_id"),
        ]
        original_message = AIMessage(
            content_blocks=original_content_blocks, tool_calls=original_tool_calls
        )

        new_tool_calls = [
            ToolCall(name="new_tool", args={"new_param": "new_value"}, id="new_id")
        ]

        result = replace_tool_calls(original_message, new_tool_calls)

        assert len(result.tool_calls) == len(new_tool_calls)
        assert result.tool_calls[0]["name"] == "new_tool"
        assert result.tool_calls[0]["args"] == {"new_param": "new_value"}
        assert result.tool_calls[0]["id"] == "new_id"

        # Check content blocks
        tool_call_blocks = [
            block for block in result.content_blocks if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "new_tool"
        assert tool_call_blocks[0]["args"] == {"new_param": "new_value"}
        assert tool_call_blocks[0]["id"] == "new_id"

    def test_replace_tool_calls_empty_list(self):
        """Test replacing with empty tool calls list."""
        original_tool_calls = [ToolCall(name="tool", args={}, id="id")]
        original_content_blocks: list[ContentBlock] = [
            create_text_block("Test message"),
            create_tool_call(name="tool", args={}, id="id"),
        ]
        original_message = AIMessage(
            content_blocks=original_content_blocks, tool_calls=original_tool_calls
        )

        new_tool_calls: list[ToolCall] = []

        result = replace_tool_calls(original_message, new_tool_calls)

        assert result.tool_calls == []

        # Check content blocks - should have no tool call blocks
        tool_call_blocks = [
            block for block in result.content_blocks if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 0

    def test_replace_tool_calls_multiple(self):
        """Test replacing with multiple tool calls."""
        original_tool_calls = [
            ToolCall(name="old_tool1", args={"param1": "value1"}, id="old_id1")
        ]
        original_content_blocks: list[ContentBlock] = [
            create_text_block("Test message"),
            create_tool_call(name="old_tool1", args={"param1": "value1"}, id="old_id1"),
        ]
        original_message = AIMessage(
            content_blocks=original_content_blocks, tool_calls=original_tool_calls
        )

        new_tool_calls = [
            ToolCall(name="new_tool1", args={"param1": "value1"}, id="new_id1"),
            ToolCall(name="new_tool2", args={"param2": "value2"}, id="new_id2"),
        ]

        result = replace_tool_calls(original_message, new_tool_calls)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "new_tool1"
        assert result.tool_calls[1]["name"] == "new_tool2"

        # Check content blocks
        tool_call_blocks = [
            block for block in result.content_blocks if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 2
        assert tool_call_blocks[0]["name"] == "new_tool1"
        assert tool_call_blocks[1]["name"] == "new_tool2"

    def test_replace_tool_calls_preserves_text_content(self):
        """Test that text content is preserved when replacing tool calls."""
        original_tool_calls = [ToolCall(name="old_tool", args={}, id="old_id")]
        original_content_blocks: list[ContentBlock] = [
            create_text_block("This is important text content"),
            create_tool_call(name="old_tool", args={}, id="old_id"),
        ]
        original_message = AIMessage(
            content_blocks=original_content_blocks, tool_calls=original_tool_calls
        )

        new_tool_calls = [ToolCall(name="new_tool", args={}, id="new_id")]

        result = replace_tool_calls(original_message, new_tool_calls)

        assert "This is important text content" in str(result.content)

        # Check content blocks
        tool_call_blocks = [
            block for block in result.content_blocks if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "new_tool"

    def test_replace_tool_calls_with_response_metadata(self):
        """Test that response metadata is preserved and updated."""
        original_tool_calls = [ToolCall(name="old_tool", args={}, id="old_id")]
        original_content_blocks: list[ContentBlock] = [
            create_text_block("Test message"),
            create_tool_call(name="old_tool", args={}, id="old_id"),
        ]
        original_metadata = {"existing_key": "existing_value"}
        original_message = AIMessage(
            content_blocks=original_content_blocks,
            tool_calls=original_tool_calls,
            response_metadata=original_metadata,
        )

        new_tool_calls = [ToolCall(name="new_tool", args={}, id="new_id")]

        result = replace_tool_calls(original_message, new_tool_calls)

        assert result.response_metadata["existing_key"] == "existing_value"
        assert result.response_metadata["output_version"] == "v1"

        # Check content blocks
        tool_call_blocks = [
            block for block in result.content_blocks if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "new_tool"

    def test_replace_tool_calls_no_original_metadata(self):
        """Test behavior when original message has no response metadata."""
        original_tool_calls = [ToolCall(name="old_tool", args={}, id="old_id")]
        original_content_blocks: list[ContentBlock] = [
            create_text_block("Test message"),
            create_tool_call(name="old_tool", args={}, id="old_id"),
        ]
        original_message = AIMessage(
            content_blocks=original_content_blocks,
            tool_calls=original_tool_calls,
            response_metadata={},
        )

        new_tool_calls = [ToolCall(name="new_tool", args={}, id="new_id")]

        result = replace_tool_calls(original_message, new_tool_calls)

        assert result.response_metadata["output_version"] == "v1"

        # Check content blocks
        tool_call_blocks = [
            block for block in result.content_blocks if block["type"] == "tool_call"
        ]
        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "new_tool"

    def test_replace_tool_calls_content_blocks(self):
        """Test that non-tool content blocks are preserved."""
        original_tool_calls = [ToolCall(name="old_tool", args={}, id="old_id")]

        text_block = create_text_block("Some text content")
        tool_call_block = create_tool_call(name="old_tool", args={}, id="old_id")

        original_message = AIMessage(
            content_blocks=[text_block, tool_call_block], tool_calls=original_tool_calls
        )

        new_tool_calls = [ToolCall(name="new_tool", args={"new": "args"}, id="new_id")]

        result = replace_tool_calls(original_message, new_tool_calls)

        # Should preserve text block and replace tool call block
        text_blocks = [
            block for block in result.content_blocks if block["type"] == "text"
        ]
        tool_call_blocks = [
            block for block in result.content_blocks if block["type"] == "tool_call"
        ]

        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Some text content"

        assert len(tool_call_blocks) == 1
        assert tool_call_blocks[0]["name"] == "new_tool"
        assert tool_call_blocks[0]["args"] == {"new": "args"}

class TestParseAttachmentIdFromUri:
    """Test URI parsing for attachment IDs."""

    def test_valid_orchestrator_uri(self):
        """Should extract UUID from valid orchestrator URI."""
        uri = "urn:uipath:cas:file:orchestrator:a940a416-b97b-4146-3089-08de5f4d0a87"

        result = parse_attachment_id_from_uri(uri)

        assert result == "a940a416-b97b-4146-3089-08de5f4d0a87"

    def test_empty_uri(self):
        """Should return None for empty URI."""
        result = parse_attachment_id_from_uri("")

        assert result is None

    def test_invalid_uri_no_uuid(self):
        """Should return None for URI without UUID."""
        result = parse_attachment_id_from_uri("urn:uipath:cas:file:orchestrator:")

        assert result is None

    def test_uri_with_uppercase_uuid_normalizes_to_lowercase(self):
        """Should normalize uppercase UUIDs to lowercase."""
        uri = "urn:uipath:cas:file:orchestrator:A940A416-B97B-4146-3089-08DE5F4D0A87"

        result = parse_attachment_id_from_uri(uri)

        assert result == "a940a416-b97b-4146-3089-08de5f4d0a87"
