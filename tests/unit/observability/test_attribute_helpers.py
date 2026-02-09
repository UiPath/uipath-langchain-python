"""Unit tests for attribute helper functions."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from uipath.tracing import AttachmentDirection, AttachmentProvider, SpanAttachment

from uipath_agents._observability.llmops.instrumentors.attribute_helpers import (
    build_task_url,
    get_tool_type_value,
    set_span_attachments,
)


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

    def test_none_returns_integration(self) -> None:
        assert get_tool_type_value(None) == "Integration"

    def test_unknown_returns_integration(self) -> None:
        assert get_tool_type_value("unknown") == "Integration"
        assert get_tool_type_value("other") == "Integration"
