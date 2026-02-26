"""Tests for hitl.py module."""

from unittest.mock import patch

from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.tools import BaseTool

from uipath_langchain.chat.hitl import (
    ARGS_MODIFIED_MESSAGE,
    CANCELLED_MESSAGE,
    ConfirmationResult,
    check_tool_confirmation,
    inject_confirmation_meta,
    request_approval,
)


class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool"

    def _run(self) -> str:
        return ""


def _make_call(args: dict | None = None) -> ToolCall:
    return ToolCall(name="mock_tool", args=args or {"query": "test"}, id="call_1")


class TestCheckToolConfirmation:
    """Tests for check_tool_confirmation."""

    def test_returns_none_when_no_metadata(self):
        """No metadata → no confirmation needed."""
        tool = MockTool()
        call = _make_call()
        assert check_tool_confirmation(call, tool) is None

    def test_returns_none_when_flag_not_set(self):
        """Metadata exists but flag is missing → no confirmation needed."""
        tool = MockTool(metadata={"other_key": True})
        call = _make_call()
        assert check_tool_confirmation(call, tool) is None

    def test_returns_none_when_flag_false(self):
        """Flag explicitly False → no confirmation needed."""
        tool = MockTool(metadata={"require_conversational_confirmation": False})
        call = _make_call()
        assert check_tool_confirmation(call, tool) is None

    @patch("uipath_langchain.chat.hitl.request_approval", return_value=None)
    def test_cancelled_returns_tool_message(self, mock_approval):
        """User rejects → ConfirmationResult with cancelled ToolMessage."""
        tool = MockTool(metadata={"require_conversational_confirmation": True})
        call = _make_call()

        result = check_tool_confirmation(call, tool)

        assert result is not None
        assert isinstance(result, ConfirmationResult)
        assert result.cancelled is not None
        assert isinstance(result.cancelled, ToolMessage)
        assert result.cancelled.content == CANCELLED_MESSAGE
        assert result.cancelled.name == "mock_tool"
        assert result.cancelled.tool_call_id == "call_1"
        assert result.args_modified is False

    @patch(
        "uipath_langchain.chat.hitl.request_approval",
        return_value={"query": "test"},
    )
    def test_approved_same_args(self, mock_approval):
        """User approves without editing → cancelled=None, args_modified=False."""
        tool = MockTool(metadata={"require_conversational_confirmation": True})
        call = _make_call({"query": "test"})

        result = check_tool_confirmation(call, tool)

        assert result is not None
        assert result.cancelled is None
        assert result.args_modified is False

    @patch(
        "uipath_langchain.chat.hitl.request_approval",
        return_value={"query": "edited"},
    )
    def test_approved_modified_args(self, mock_approval):
        """User edits args → cancelled=None, args_modified=True, call updated."""
        tool = MockTool(metadata={"require_conversational_confirmation": True})
        call = _make_call({"query": "original"})

        result = check_tool_confirmation(call, tool)

        assert result is not None
        assert result.cancelled is None
        assert result.args_modified is True
        assert call["args"] == {"query": "edited"}


class TestInjectConfirmationMeta:
    """Tests for inject_confirmation_meta."""

    def test_injects_meta_into_content(self):
        msg = ToolMessage(content="some result", name="t", tool_call_id="c1")
        inject_confirmation_meta(msg, ARGS_MODIFIED_MESSAGE)
        assert ARGS_MODIFIED_MESSAGE in msg.content
        assert "some result" in msg.content

    def test_wraps_content_as_json(self):
        msg = ToolMessage(content="hello", name="t", tool_call_id="c1")
        inject_confirmation_meta(msg, "test meta")
        assert msg.content == '{"meta": "test meta", "result": hello}'


class TestRequestApprovalTruthiness:
    """Tests for the truthiness fix in request_approval."""

    @patch("uipath_langchain.chat.hitl.interrupt")
    def test_empty_dict_input_preserved(self, mock_interrupt):
        """Empty dict from user edits should not be replaced by original args."""
        mock_interrupt.return_value = {"value": {"approved": True, "input": {}}}
        tool = MockTool()
        result = request_approval({"query": "test", "tool_call_id": "c1"}, tool)
        assert result == {}

    @patch("uipath_langchain.chat.hitl.interrupt")
    def test_empty_list_input_preserved(self, mock_interrupt):
        """Empty list from user edits should not be replaced by original args."""
        mock_interrupt.return_value = {"value": {"approved": True, "input": []}}
        tool = MockTool()
        result = request_approval({"query": "test", "tool_call_id": "c1"}, tool)
        assert result == []

    @patch("uipath_langchain.chat.hitl.interrupt")
    def test_none_input_falls_back_to_original(self, mock_interrupt):
        """None input should fall back to original tool_args."""
        mock_interrupt.return_value = {"value": {"approved": True, "input": None}}
        tool = MockTool()
        result = request_approval({"query": "test", "tool_call_id": "c1"}, tool)
        assert result == {"query": "test"}

    @patch("uipath_langchain.chat.hitl.interrupt")
    def test_missing_input_falls_back_to_original(self, mock_interrupt):
        """Missing input key should fall back to original tool_args."""
        mock_interrupt.return_value = {"value": {"approved": True}}
        tool = MockTool()
        result = request_approval({"query": "test", "tool_call_id": "c1"}, tool)
        assert result == {"query": "test"}

    @patch("uipath_langchain.chat.hitl.interrupt")
    def test_rejected_returns_none(self, mock_interrupt):
        """Rejected approval returns None."""
        mock_interrupt.return_value = {"value": {"approved": False}}
        tool = MockTool()
        result = request_approval({"query": "test", "tool_call_id": "c1"}, tool)
        assert result is None
