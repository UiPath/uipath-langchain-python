"""Unit tests for attribute helper functions."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from uipath.tracing import AttachmentDirection, AttachmentProvider, SpanAttachment

from uipath_agents._observability.llmops.instrumentors.attribute_helpers import (
    build_task_url,
    filter_output,
    get_tool_type_value,
    sanitize_file_data,
    set_span_attachments,
    set_tool_result,
)


class TestSanitizeFileData:
    """Tests for sanitize_file_data function."""

    def test_returns_plain_string_unchanged(self) -> None:
        """Test that normal strings pass through without modification."""
        assert sanitize_file_data("hello world") == "hello world"

    def test_returns_short_string_unchanged(self) -> None:
        """Test that short strings are never treated as base64."""
        short_b64 = "SGVsbG8gV29ybGQ="
        assert sanitize_file_data(short_b64) == short_b64

    def test_replaces_data_uri(self) -> None:
        """Test that data URIs with base64 encoding are replaced."""
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        assert sanitize_file_data(data_uri) == "<base64 data omitted>"

    def test_replaces_long_base64_string(self) -> None:
        """Test that long strings matching base64 pattern are replaced."""
        long_b64 = "A" * 1500
        assert sanitize_file_data(long_b64) == "<base64 data omitted>"

    def test_keeps_long_non_base64_string(self) -> None:
        """Test that long strings with non-base64 chars are kept."""
        long_text = "Hello, this is a normal sentence! " * 50
        assert sanitize_file_data(long_text) == long_text

    def test_replaces_bytes(self) -> None:
        """Test that bytes objects are replaced with size placeholder."""
        data = b"\x89PNG\r\n\x1a\n" * 100
        result = sanitize_file_data(data)
        assert result == f"<bytes: {len(data)} bytes>"

    def test_sanitizes_list_recursively(self) -> None:
        """Test that lists are recursively sanitized."""
        content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": "A" * 1500},
        ]
        result = sanitize_file_data(content)
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "Describe this image"}
        assert result[1]["image_url"] == "<base64 data omitted>"

    def test_sanitizes_dict_data_key_with_long_base64(self) -> None:
        """Test that dict 'data' key with long base64 value is replaced."""
        obj = {"data": "A" * 1500, "name": "file.png"}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["data"] == "<base64 data omitted>"
        assert result["name"] == "file.png"

    def test_sanitizes_dict_bytes_key(self) -> None:
        """Test that dict 'bytes' key with bytes value is replaced."""
        obj = {"bytes": b"\x00" * 500, "mime_type": "image/png"}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["bytes"] == "<bytes: 500 bytes>"
        assert result["mime_type"] == "image/png"

    def test_sanitizes_dict_file_data_key(self) -> None:
        """Test that dict 'file_data' key with long base64 is replaced."""
        obj = {"file_data": "B" * 2000}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["file_data"] == "<base64 data omitted>"

    def test_sanitizes_dict_image_url_key_with_long_base64(self) -> None:
        """Test that dict 'image_url' key with long base64 value is replaced."""
        obj = {"image_url": "A" * 2000}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["image_url"] == "<base64 data omitted>"

    def test_sanitizes_data_uri_under_image_url_key(self) -> None:
        """Test that data URI string directly under image_url key is replaced."""
        obj = {"image_url": "data:image/jpeg;base64," + "A" * 5000}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["image_url"] == "<base64 data omitted>"

    def test_sanitizes_data_uri_under_file_data_key(self) -> None:
        """Test that data URI under file_data key is replaced."""
        obj = {"file_data": "data:application/pdf;base64," + "B" * 10000}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["file_data"] == "<base64 data omitted>"

    def test_sanitizes_data_uri_under_data_key(self) -> None:
        """Test that data URI under data key is replaced."""
        obj = {"data": "data:image/png;base64," + "C" * 3000}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["data"] == "<base64 data omitted>"

    def test_sanitizes_openai_input_image_content_block(self) -> None:
        """Test OpenAI input_image content block format."""
        content = [
            {"type": "text", "text": "Describe this"},
            {
                "type": "input_image",
                "source": {
                    "type": "base64",
                    "data": "A" * 1_000_000,
                    "media_type": "image/png",
                },
            },
        ]
        result = sanitize_file_data(content)
        assert isinstance(result, list)
        result_str = str(result)
        assert len(result_str) < 300
        assert "Describe this" in result_str

    def test_sanitizes_openai_file_content_with_data_uri(self) -> None:
        """Test OpenAI file content with data URI in file_data."""
        content = [
            {"type": "text", "text": "Summarize this PDF"},
            {
                "type": "file",
                "file": {
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64," + "D" * 2_000_000,
                },
            },
        ]
        result = sanitize_file_data(content)
        assert isinstance(result, list)
        result_str = str(result)
        assert len(result_str) < 500
        assert "Summarize this PDF" in result_str
        assert "report.pdf" in result_str

    def test_keeps_dict_data_key_with_short_string(self) -> None:
        """Test that dict 'data' key with short non-base64 value is kept."""
        obj = {"data": "short value"}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["data"] == "short value"

    def test_keeps_dict_data_key_when_value_is_dict(self) -> None:
        """Test that 'data' key with dict value recurses normally."""
        obj = {"data": {"nested": "data:image/png;base64,abc123"}}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["data"]["nested"] == "<base64 data omitted>"

    def test_recurses_into_nested_dicts(self) -> None:
        """Test deep nesting is handled."""
        obj = {"outer": {"inner": {"data": "C" * 1500}}}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["outer"]["inner"]["data"] == "<base64 data omitted>"

    def test_passes_through_other_types(self) -> None:
        """Test that non-str/bytes/list/dict types pass through."""
        assert sanitize_file_data(42) == 42
        assert sanitize_file_data(3.14) == 3.14
        assert sanitize_file_data(None) is None
        assert sanitize_file_data(True) is True

    def test_sanitizes_list_inside_special_dict_key(self) -> None:
        """Test that list values under special keys are recursively sanitized."""
        obj = {"data": [b"\x00" * 100, "normal"]}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["data"] == ["<bytes: 100 bytes>", "normal"]

    def test_multimodal_message_content(self) -> None:
        """Test realistic multimodal LLM message content structure."""
        content = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 5000},
            },
        ]
        result = sanitize_file_data(content)
        assert isinstance(result, list)
        assert result[0]["text"] == "What is in this image?"
        assert result[1]["image_url"]["url"] == "<base64 data omitted>"

    def test_no_large_blob_survives_multi_mb_data_uri(self) -> None:
        """Test that a 5MB data URI in any position is replaced."""
        huge_payload = "data:application/pdf;base64," + "A" * (5 * 1024 * 1024)
        result = sanitize_file_data(huge_payload)
        assert len(str(result)) < 100

    def test_no_large_blob_survives_raw_base64(self) -> None:
        """Test that a 2MB raw base64 string is replaced."""
        huge_b64 = "A" * (2 * 1024 * 1024)
        result = sanitize_file_data(huge_b64)
        assert len(str(result)) < 100

    def test_no_large_blob_survives_bytes_value(self) -> None:
        """Test that a 10MB bytes value is replaced with a small placeholder."""
        huge_bytes = b"\x00" * (10 * 1024 * 1024)
        result = sanitize_file_data(huge_bytes)
        assert len(str(result)) < 100

    def test_no_large_blob_survives_nested_in_dict(self) -> None:
        """Test that large data nested anywhere in a dict is caught."""
        obj = {
            "metadata": {"name": "report.pdf"},
            "content": {
                "pages": [
                    {"data": "B" * (1 * 1024 * 1024)},
                    {"image_url": {"url": "data:image/png;base64," + "C" * 500_000}},
                ]
            },
        }
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        result_str = str(result)
        assert len(result_str) < 500
        assert "report.pdf" in result_str

    def test_no_large_blob_survives_in_list_of_messages(self) -> None:
        """Test realistic LLM multimodal messages with multiple large images."""
        messages = [
            {"type": "text", "text": "Compare these images"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + "A" * 1_000_000},
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + "B" * 2_000_000},
            },
            {"type": "text", "text": "Which is better?"},
        ]
        result = sanitize_file_data(messages)
        assert isinstance(result, list)
        result_str = str(result)
        assert len(result_str) < 500
        assert "Compare these images" in result_str
        assert "Which is better?" in result_str

    def test_no_large_blob_survives_file_data_key(self) -> None:
        """Test that file_data key with multi-MB value is replaced."""
        obj = {"file_data": "D" * (3 * 1024 * 1024), "filename": "scan.pdf"}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert len(str(result)) < 200

    def test_no_large_blob_survives_bytes_key_in_dict(self) -> None:
        """Test that bytes key with large bytes value is replaced."""
        obj = {"bytes": b"\xff" * (5 * 1024 * 1024), "type": "image/png"}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert len(str(result)) < 200

    def test_base64_with_url_safe_chars(self) -> None:
        """Test that URL-safe base64 (with - and _) is also caught."""
        url_safe_b64 = "A-B_C+D/" * 200
        assert len(url_safe_b64) > 1000
        result = sanitize_file_data(url_safe_b64)
        assert result == "<base64 data omitted>"

    def test_empty_structures_pass_through(self) -> None:
        """Test that empty collections are not affected."""
        assert sanitize_file_data([]) == []
        assert sanitize_file_data({}) == {}
        assert sanitize_file_data("") == ""

    # --- File-data key targeting ---

    def test_file_data_key_with_int_value_preserved(self) -> None:
        """File-data keys with non-string/non-bytes values pass through unchanged."""
        obj = {"data": 42, "bytes": 0, "file_data": True}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result == {"data": 42, "bytes": 0, "file_data": True}

    def test_file_data_key_with_none_value_preserved(self) -> None:
        """File-data keys with None value pass through unchanged."""
        obj = {"data": None, "image_url": None}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result == {"data": None, "image_url": None}

    def test_file_data_key_with_dict_value_recurses(self) -> None:
        """File-data keys with dict values recurse into the dict (not treated as leaf)."""
        obj = {"image_url": {"url": "data:image/png;base64," + "A" * 5000}}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["image_url"]["url"] == "<base64 data omitted>"

    def test_non_file_key_with_data_uri_string_sanitized(self) -> None:
        """Non-file keys with data URI strings are still sanitized via recursion."""
        obj = {"description": "data:image/png;base64," + "A" * 5000}
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["description"] == "<base64 data omitted>"

    def test_non_file_key_with_nested_file_data(self) -> None:
        """Non-file keys with nested dicts containing file data are recursed into."""
        obj = {
            "metadata": {
                "image": {"data": "A" * 2000},
            }
        }
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["metadata"]["image"]["data"] == "<base64 data omitted>"

    def test_all_four_file_data_keys_recognized(self) -> None:
        """All four file-data keys (data, bytes, file_data, image_url) are handled."""
        obj = {
            "data": "data:image/png;base64," + "A" * 100,
            "bytes": b"\x00" * 500,
            "file_data": "B" * 2000,
            "image_url": "C" * 2000,
        }
        result = sanitize_file_data(obj)
        assert isinstance(result, dict)
        assert result["data"] == "<base64 data omitted>"
        assert result["bytes"] == "<bytes: 500 bytes>"
        assert result["file_data"] == "<base64 data omitted>"
        assert result["image_url"] == "<base64 data omitted>"


class TestSetSpanAttachments:
    """Tests for set_span_attachments function."""

    def _create_mock_attachments(
        self, attachment_id: str, direction: AttachmentDirection
    ) -> list[SpanAttachment]:
        """Create mock SpanAttachment objects for testing."""
        return [
            SpanAttachment(
                id=attachment_id,
                file_name="test.pdf",
                mime_type="application/pdf",
                provider=AttachmentProvider.ORCHESTRATOR,
                direction=direction,
            )
        ]

    @pytest.fixture
    def mock_span(self) -> MagicMock:
        """Create a mock Span object."""
        span = MagicMock()
        span.attributes = {}
        return span

    @pytest.fixture
    def sample_output_schema(self) -> dict[str, Any]:
        """Sample output schema with job attachments."""
        return {
            "type": "object",
            "properties": {
                "result": {"type": "string"},
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "full_name": {"type": "string"},
                            "mime_type": {"type": "string"},
                        },
                    },
                },
            },
        }

    @pytest.fixture
    def sample_output_with_attachments(self) -> dict[str, Any]:
        """Sample output data containing attachments."""
        return {
            "result": "success",
            "attachments": [
                {
                    "id": "att-123",
                    "full_name": "document.pdf",
                    "mime_type": "application/pdf",
                }
            ],
        }

    @patch(
        "uipath_agents._observability.llmops.instrumentors.attribute_helpers.get_span_attachments"
    )
    def test_set_attachments_on_new_span_object(
        self,
        mock_get_attachments: MagicMock,
        mock_span: MagicMock,
        sample_output_with_attachments: dict[str, Any],
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Test setting attachments on a Span object without existing attachments."""

        mock_get_attachments.return_value = self._create_mock_attachments(
            "att-123", AttachmentDirection.OUT
        )

        set_span_attachments(
            mock_span,
            sample_output_with_attachments,
            sample_output_schema,
            AttachmentDirection.OUT,
        )

        assert mock_span.set_attribute.called
        call_args = mock_span.set_attribute.call_args
        assert call_args[0][0] == "attachments"

        attachments_json = call_args[0][1]
        attachments = json.loads(attachments_json)
        assert isinstance(attachments, list)
        assert len(attachments) > 0
        assert attachments[0]["direction"] == AttachmentDirection.OUT

    @patch(
        "uipath_agents._observability.llmops.instrumentors.attribute_helpers.get_span_attachments"
    )
    def test_set_attachments_on_span_dict(
        self,
        mock_get_attachments: MagicMock,
        sample_output_with_attachments: dict[str, Any],
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Test setting attachments on a span dict (upsert scenario)."""

        mock_get_attachments.return_value = self._create_mock_attachments(
            "att-456", AttachmentDirection.OUT
        )

        span_dict: dict[str, Any] = {"attributes": {}}

        set_span_attachments(
            span_dict,
            sample_output_with_attachments,
            sample_output_schema,
            AttachmentDirection.OUT,
        )

        assert "attachments" in span_dict["attributes"]
        attachments_json = span_dict["attributes"]["attachments"]
        attachments = json.loads(attachments_json)
        assert isinstance(attachments, list)
        assert len(attachments) > 0

    @patch(
        "uipath_agents._observability.llmops.instrumentors.attribute_helpers.get_span_attachments"
    )
    def test_merge_attachments_on_span_object(
        self,
        mock_get_attachments: MagicMock,
        mock_span: MagicMock,
        sample_output_with_attachments: dict[str, Any],
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Test merging new attachments with existing ones on Span object."""
        existing_attachment = SpanAttachment(
            id="existing-123",
            file_name="existing.txt",
            mime_type="text/plain",
            provider=AttachmentProvider.ORCHESTRATOR,
            direction=AttachmentDirection.IN,
        )
        existing_json = json.dumps([existing_attachment.model_dump(by_alias=True)])
        mock_span.attributes = {"attachments": existing_json}

        mock_get_attachments.return_value = self._create_mock_attachments(
            "new-123", AttachmentDirection.OUT
        )

        set_span_attachments(
            mock_span,
            sample_output_with_attachments,
            sample_output_schema,
            AttachmentDirection.OUT,
        )

        call_args = mock_span.set_attribute.call_args
        attachments_json = call_args[0][1]
        attachments = json.loads(attachments_json)

        assert len(attachments) == 2
        assert attachments[0]["direction"] == AttachmentDirection.IN
        assert attachments[1]["direction"] == AttachmentDirection.OUT

    @patch(
        "uipath_agents._observability.llmops.instrumentors.attribute_helpers.get_span_attachments"
    )
    def test_merge_attachments_on_span_dict(
        self,
        mock_get_attachments: MagicMock,
        sample_output_with_attachments: dict[str, Any],
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Test merging new attachments with existing ones on span dict."""

        existing_attachment = {
            "id": "existing-456",
            "fileName": "existing.docx",
            "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "provider": AttachmentProvider.ORCHESTRATOR,
            "direction": AttachmentDirection.IN,
        }
        span_dict: dict[str, Any] = {
            "attributes": {"attachments": json.dumps([existing_attachment])}
        }

        mock_get_attachments.return_value = self._create_mock_attachments(
            "new-456", AttachmentDirection.OUT
        )

        set_span_attachments(
            span_dict,
            sample_output_with_attachments,
            sample_output_schema,
            AttachmentDirection.OUT,
        )

        attachments_json = span_dict["attributes"]["attachments"]
        attachments = json.loads(attachments_json)
        assert len(attachments) == 2
        assert attachments[0] == existing_attachment
        assert attachments[1]["direction"] == AttachmentDirection.OUT

    def test_no_attachments_when_output_is_none(
        self, mock_span: MagicMock, sample_output_schema: dict[str, Any]
    ) -> None:
        """Test that no attachments are set when output is None."""
        set_span_attachments(
            mock_span, None, sample_output_schema, AttachmentDirection.OUT
        )

        mock_span.set_attribute.assert_not_called()

    def test_no_attachments_when_schema_is_none(
        self, mock_span: MagicMock, sample_output_with_attachments: dict[str, Any]
    ) -> None:
        """Test that no attachments are set when schema is None."""
        set_span_attachments(
            mock_span, sample_output_with_attachments, None, AttachmentDirection.OUT
        )

        mock_span.set_attribute.assert_not_called()

    def test_no_attachments_when_output_not_dict(
        self, mock_span: MagicMock, sample_output_schema: dict[str, Any]
    ) -> None:
        """Test that no attachments are set when output is not a dict."""
        set_span_attachments(
            mock_span, "string output", sample_output_schema, AttachmentDirection.OUT
        )

        mock_span.set_attribute.assert_not_called()

    @patch(
        "uipath_agents._observability.llmops.instrumentors.attribute_helpers.get_span_attachments"
    )
    def test_handles_invalid_existing_attachments_json(
        self,
        mock_get_attachments: MagicMock,
        mock_span: MagicMock,
        sample_output_with_attachments: dict[str, Any],
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Test that invalid JSON in existing attachments is handled gracefully."""

        mock_span.attributes = {"attachments": "invalid json {"}
        mock_get_attachments.return_value = self._create_mock_attachments(
            "recovery-123", AttachmentDirection.OUT
        )

        set_span_attachments(
            mock_span,
            sample_output_with_attachments,
            sample_output_schema,
            AttachmentDirection.OUT,
        )

        assert mock_span.set_attribute.called
        call_args = mock_span.set_attribute.call_args
        attachments_json = call_args[0][1]
        attachments = json.loads(attachments_json)
        assert isinstance(attachments, list)
        assert len(attachments) > 0

    @patch(
        "uipath_agents._observability.llmops.instrumentors.attribute_helpers.get_span_attachments"
    )
    def test_span_dict_without_attributes_key(
        self,
        mock_get_attachments: MagicMock,
        sample_output_with_attachments: dict[str, Any],
        sample_output_schema: dict[str, Any],
    ) -> None:
        """Test setting attachments on span dict without attributes key."""

        mock_get_attachments.return_value = self._create_mock_attachments(
            "new-dict-123", AttachmentDirection.OUT
        )

        span_dict: dict[str, Any] = {}

        set_span_attachments(
            span_dict,
            sample_output_with_attachments,
            sample_output_schema,
            AttachmentDirection.OUT,
        )

        assert "attributes" in span_dict
        assert "attachments" in span_dict["attributes"]

    @patch(
        "uipath_agents._observability.llmops.instrumentors.attribute_helpers.get_span_attachments"
    )
    def test_no_attachments_extracted_from_output(
        self, mock_get_attachments: MagicMock, mock_span: MagicMock
    ) -> None:
        """Test when output has no extractable attachments."""

        mock_get_attachments.return_value = None

        output = {"result": "success"}
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }

        set_span_attachments(mock_span, output, schema, AttachmentDirection.OUT)
        mock_span.set_attribute.assert_not_called()


class TestBuildTaskUrl:
    """Tests for build_task_url function.

    UIPATH_URL already includes org/tenant in the path,
    so build_task_url only needs UIPATH_URL.
    """

    def test_builds_url_with_uipath_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UIPATH_URL", "https://alpha.uipath.com/org-123/tenant-456")

        url = build_task_url(12345)

        expected = "https://alpha.uipath.com/org-123/tenant-456/actions_/tasks/12345"
        assert url == expected

    def test_strips_trailing_slash_from_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UIPATH_URL", "https://alpha.uipath.com/org/tenant/")

        url = build_task_url(1)

        assert url == "https://alpha.uipath.com/org/tenant/actions_/tasks/1"

    def test_returns_none_when_uipath_url_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("UIPATH_URL", raising=False)

        assert build_task_url(123) is None

    def test_accepts_string_task_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UIPATH_URL", "https://cloud.uipath.com/org/tenant")

        url = build_task_url("task-abc-123")

        expected = "https://cloud.uipath.com/org/tenant/actions_/tasks/task-abc-123"
        assert url == expected


class TestGetToolTypeValue:
    """Tests for get_tool_type_value function."""

    def test_escalation_returns_escalation(self) -> None:
        assert get_tool_type_value("escalation") == "Escalation"

    def test_agent_returns_agent(self) -> None:
        assert get_tool_type_value("agent") == "Agent"

    def test_process_returns_process(self) -> None:
        assert get_tool_type_value("process") == "Process"

    def test_api_returns_api(self) -> None:
        assert get_tool_type_value("api") == "Api"

    def test_processorchestration_returns_agentic_process(self) -> None:
        assert get_tool_type_value("processorchestration") == "agenticProcess"

    def test_mcp_returns_mcp(self) -> None:
        assert get_tool_type_value("mcp") == "Mcp"

    def test_internal_returns_internal(self) -> None:
        assert get_tool_type_value("internal") == "Internal"

    def test_none_returns_integration(self) -> None:
        assert get_tool_type_value(None) == "Integration"

    def test_unknown_returns_integration(self) -> None:
        assert get_tool_type_value("unknown") == "Integration"
        assert get_tool_type_value("other") == "Integration"


class TestSetToolResult:
    """Tests for set_tool_result function."""

    def test_sets_dict_result_as_json(self) -> None:
        span = MagicMock()
        set_tool_result(span, {"key": "value"})
        span.set_attribute.assert_called_once()
        name, value = span.set_attribute.call_args[0]
        assert name == "result"
        assert json.loads(value) == {"key": "value"}

    def test_sets_list_result_as_json(self) -> None:
        span = MagicMock()
        set_tool_result(span, [{"type": "text", "text": "hi"}])
        span.set_attribute.assert_called_once()
        name, value = span.set_attribute.call_args[0]
        assert name == "result"
        assert json.loads(value) == [{"type": "text", "text": "hi"}]

    def test_sets_string_result_as_str(self) -> None:
        span = MagicMock()
        set_tool_result(span, "plain text")
        span.set_attribute.assert_called_once_with("result", "plain text")

    def test_skips_none_output(self) -> None:
        span = MagicMock()
        set_tool_result(span, None)
        span.set_attribute.assert_not_called()

    def test_custom_attribute_name(self) -> None:
        span = MagicMock()
        set_tool_result(span, {"data": 1}, "output")
        name, _ = span.set_attribute.call_args[0]
        assert name == "output"

    def test_skips_no_content_dict(self) -> None:
        span = MagicMock()
        set_tool_result(span, {"status": "completed", "__internal": "NO_CONTENT"})
        span.set_attribute.assert_not_called()

    def test_skips_no_content_string(self) -> None:
        span = MagicMock()
        set_tool_result(span, '{"status": "completed", "__internal": "NO_CONTENT"}')
        span.set_attribute.assert_not_called()


class TestFilterOutput:
    """Tests for filter_output function."""

    def test_returns_none_for_none(self) -> None:
        assert filter_output(None) is None

    def test_returns_none_for_no_content_dict(self) -> None:
        marker = {"status": "completed", "__internal": "NO_CONTENT"}
        assert filter_output(marker) is None

    def test_returns_none_for_no_content_string(self) -> None:
        marker = '{"status": "completed", "__internal": "NO_CONTENT"}'
        assert filter_output(marker) is None

    def test_passes_through_normal_dict(self) -> None:
        output = {"key": "value"}
        assert filter_output(output) == {"key": "value"}

    def test_passes_through_normal_string(self) -> None:
        assert filter_output("hello") == "hello"

    def test_passes_through_list(self) -> None:
        output = [1, 2, 3]
        assert filter_output(output) == [1, 2, 3]

    def test_passes_through_dict_with_internal_key_but_different_value(self) -> None:
        output = {"__internal": "SOME_OTHER_VALUE"}
        assert filter_output(output) == {"__internal": "SOME_OTHER_VALUE"}
