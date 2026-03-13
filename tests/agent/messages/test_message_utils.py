"""Tests for agent/messages/message_utils.py module."""

from langchain.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.messages.content import (
    ContentBlock,
    create_text_block,
    create_tool_call,
)
from langgraph.graph.message import add_messages

from uipath_langchain.agent.messages.message_utils import replace_tool_calls


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

    def test_replace_tool_calls_preserves_message_id(self):
        """Test that the original message id is preserved after replacement."""
        original_tool_calls = [ToolCall(name="old_tool", args={}, id="old_id")]
        original_content_blocks: list[ContentBlock] = [
            create_text_block("Test message"),
            create_tool_call(name="old_tool", args={}, id="old_id"),
        ]
        original_message = AIMessage(
            content_blocks=original_content_blocks,
            tool_calls=original_tool_calls,
            id="msg-original-id",
        )

        new_tool_calls = [ToolCall(name="new_tool", args={}, id="new_id")]

        result = replace_tool_calls(original_message, new_tool_calls)

        assert result.id == "msg-original-id"

    def test_replace_tool_calls_updated_args_visible_via_add_messages(self):
        """Test that updated tool call args are visible after add_messages processes them.

        Reproduces the HITL bug: when a human reviews and updates activity input
        during an escalation, the activity must execute with the reviewed args.
        Without id preservation, add_messages appends a duplicate AIMessage
        instead of replacing the original, causing the tool to run with stale args.
        """
        original_tool_calls = [
            ToolCall(
                name="my_activity", args={"input": "original_value"}, id="call_1"
            )
        ]
        original_ai_message = AIMessage(
            content_blocks=[
                create_text_block("I will invoke the activity"),
                create_tool_call(
                    name="my_activity", args={"input": "original_value"}, id="call_1"
                ),
            ],
            tool_calls=original_tool_calls,
            id="msg-from-llm",
        )

        messages: list = [
            HumanMessage(content="do something", id="msg-human"),
            original_ai_message,
        ]

        # Simulate HITL review: human changes the input
        reviewed_tool_calls = [
            ToolCall(
                name="my_activity", args={"input": "reviewed_value"}, id="call_1"
            )
        ]
        updated_ai_message = replace_tool_calls(original_ai_message, reviewed_tool_calls)

        # Simulate what Command(update={"messages": [updated_ai_message]}) does
        result_messages = add_messages(messages, [updated_ai_message])

        # There must be exactly one AIMessage — not a duplicate
        ai_messages = [m for m in result_messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1, (
            f"Expected 1 AIMessage but got {len(ai_messages)}; "
            "add_messages appended instead of replacing (id mismatch)"
        )

        # The surviving AIMessage must carry the reviewed args
        assert ai_messages[0].tool_calls[0]["args"] == {"input": "reviewed_value"}

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
