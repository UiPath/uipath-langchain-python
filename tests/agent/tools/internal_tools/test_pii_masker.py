"""Tests for pii_masker.py module."""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest
from uipath.core.feature_flags import FeatureFlags
from uipath.platform.semantic_proxy import (
    PiiDetectionResponse,
    PiiDocumentResult,
    PiiEntity,
    PiiEntityThreshold,
    PiiFileResult,
)

from uipath_langchain.agent.multimodal import FileInfo
from uipath_langchain.agent.tools.internal_tools.pii_masker import (
    _FEATURE_FLAG,
    PiiMasker,
)


@pytest.fixture
def reset_feature_flags():
    """Reset FeatureFlags registry before and after each test."""
    FeatureFlags.reset_flags()
    yield
    FeatureFlags.reset_flags()


def _make_pii_response(
    *,
    masked_prompt: str = "original",
    entities: list[PiiEntity] | None = None,
    redacted_files: list[PiiFileResult] | None = None,
) -> PiiDetectionResponse:
    return PiiDetectionResponse(
        response=[
            PiiDocumentResult(
                id="user-prompt",
                role="user",
                masked_document=masked_prompt,
                initial_document="original",
                pii_entities=entities or [],
            )
        ],
        files=redacted_files or [],
    )


def _make_client(
    response: PiiDetectionResponse,
    upload_result: uuid.UUID | None = None,
) -> Mock:
    client = Mock()
    client.semantic_proxy = Mock()
    client.semantic_proxy.detect_pii_async = AsyncMock(return_value=response)
    client.attachments = Mock()
    client.attachments.upload_async = AsyncMock(
        return_value=upload_result or uuid.uuid4()
    )
    return client


class TestIsPolicyEnabled:
    """Test cases for PiiMasker.is_policy_enabled."""

    def test_returns_false_when_feature_flag_disabled(self, reset_feature_flags):
        FeatureFlags.configure_flags({_FEATURE_FLAG: False})
        policy = {"data": {"container": {"pii-in-flight-agents": True}}}

        assert PiiMasker.is_policy_enabled(policy) is False

    def test_returns_false_when_policy_is_none(self, reset_feature_flags):
        assert PiiMasker.is_policy_enabled(None) is False

    def test_returns_false_when_policy_missing_gate(self, reset_feature_flags):
        assert PiiMasker.is_policy_enabled({"data": {"container": {}}}) is False

    def test_returns_false_when_gate_is_false(self, reset_feature_flags):
        policy = {"data": {"container": {"pii-in-flight-agents": False}}}
        assert PiiMasker.is_policy_enabled(policy) is False

    def test_returns_true_when_flag_enabled_and_policy_enables(
        self, reset_feature_flags
    ):
        FeatureFlags.configure_flags({_FEATURE_FLAG: True})
        policy = {"data": {"container": {"pii-in-flight-agents": True}}}
        assert PiiMasker.is_policy_enabled(policy) is True


class TestEntityThresholdsFromPolicy:
    """Test cases for PiiMasker._entity_thresholds_from_policy."""

    def test_returns_empty_for_none_policy(self):
        assert PiiMasker(Mock(), None)._entity_thresholds_from_policy() == []

    def test_returns_empty_when_table_missing(self):
        masker = PiiMasker(Mock(), {"data": {}})
        assert masker._entity_thresholds_from_policy() == []

    def test_filters_disabled_entries(self):
        policy = {
            "data": {
                "pii-entity-table": [
                    {
                        "pii-entity-is-enabled": False,
                        "pii-entity-category": "Email",
                        "pii-entity-confidence-threshold": 0.5,
                    },
                ]
            }
        }
        assert PiiMasker(Mock(), policy)._entity_thresholds_from_policy() == []

    def test_filters_entries_with_missing_category_or_confidence(self):
        policy = {
            "data": {
                "pii-entity-table": [
                    {
                        "pii-entity-is-enabled": True,
                        "pii-entity-confidence-threshold": 0.8,
                    },
                    {
                        "pii-entity-is-enabled": True,
                        "pii-entity-category": "SSN",
                    },
                ]
            }
        }
        assert PiiMasker(Mock(), policy)._entity_thresholds_from_policy() == []

    def test_returns_enabled_entries_as_thresholds(self):
        policy = {
            "data": {
                "pii-entity-table": [
                    {
                        "pii-entity-is-enabled": True,
                        "pii-entity-category": "Email",
                        "pii-entity-confidence-threshold": 0.5,
                    },
                    {
                        "pii-entity-is-enabled": False,
                        "pii-entity-category": "Phone",
                        "pii-entity-confidence-threshold": 0.7,
                    },
                    {
                        "pii-entity-is-enabled": True,
                        "pii-entity-category": "SSN",
                        "pii-entity-confidence-threshold": 0.9,
                    },
                ]
            }
        }

        thresholds = PiiMasker(Mock(), policy)._entity_thresholds_from_policy()

        assert thresholds == [
            PiiEntityThreshold(category="Email", confidence_threshold=0.5),
            PiiEntityThreshold(category="SSN", confidence_threshold=0.9),
        ]


class TestWithMaskedUrl:
    """Test cases for PiiMasker._with_masked_url."""

    def test_preserves_url_and_name_and_carries_masked_url(self):
        original = FileInfo(
            url="https://orig/report.pdf",
            name="report.pdf",
            mime_type="application/pdf",
            attachment_id="att-uuid",
        )

        annotated = PiiMasker._with_masked_url(original, "https://redacted/report.pdf")

        assert annotated.url == "https://orig/report.pdf"
        assert annotated.name == "report.pdf"
        assert annotated.mime_type == "application/pdf"
        assert annotated.masked_attachment_url == "https://redacted/report.pdf"
        assert annotated.attachment_id == "att-uuid"

    def test_preserves_fields_when_attachment_id_missing(self):
        original = FileInfo(
            url="https://orig/readme", name="readme", mime_type="text/plain"
        )

        annotated = PiiMasker._with_masked_url(original, "https://redacted/readme")

        assert annotated.url == "https://orig/readme"
        assert annotated.name == "readme"
        assert annotated.masked_attachment_url == "https://redacted/readme"
        assert annotated.attachment_id is None


class TestApply:
    """Test cases for PiiMasker.apply."""

    async def test_masks_prompt_and_annotates_files(self, httpx_mock):
        entity = PiiEntity(
            pii_text="john@example.com",
            replacement_text="[EMAIL]",
            pii_type="Email",
            offset=10,
            confidence_score=0.95,
        )
        redacted_file = PiiFileResult(
            file_name="doc.pdf",
            file_url="https://redacted/doc.pdf",
            pii_entities=[entity],
        )
        response = _make_pii_response(
            masked_prompt="Please contact [EMAIL] for info",
            entities=[entity],
            redacted_files=[redacted_file],
        )
        uploaded_uuid = uuid.uuid4()
        client = _make_client(response, upload_result=uploaded_uuid)
        files = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
                attachment_id="orig-uuid",
            )
        ]
        httpx_mock.add_response(
            url="https://redacted/doc.pdf", content=b"redacted-bytes"
        )

        masker = PiiMasker(client, policy=None)
        masked_prompt, masked_files = await masker.apply(
            "Please contact john@example.com for info", files
        )

        assert masked_prompt == "Please contact [EMAIL] for info"
        assert len(masked_files) == 1
        assert masked_files[0].url == "https://orig/doc.pdf"
        assert masked_files[0].name == "doc.pdf"
        assert masked_files[0].masked_attachment_url == "https://redacted/doc.pdf"
        assert masked_files[0].masked_attachment_id == str(uploaded_uuid)
        assert masked_files[0].attachment_id == "orig-uuid"

        request = client.semantic_proxy.detect_pii_async.call_args[0][0]
        assert request.files[0].file_name == "doc.pdf"
        assert request.files[0].file_type == "pdf"

        upload_call = client.attachments.upload_async.call_args
        assert upload_call.kwargs["name"] == "pii_masked_doc.pdf"
        assert upload_call.kwargs["content"] == b"redacted-bytes"

    async def test_returns_original_files_when_no_redactions(self):
        response = _make_pii_response(masked_prompt="clean prompt")
        client = _make_client(response)
        files = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
            )
        ]

        _, masked_files = await PiiMasker(client, None).apply("clean prompt", files)

        # Same list instance returned when no redactions happened.
        assert masked_files is files

    async def test_unmatched_redacted_filename_falls_back_to_original_url(
        self, httpx_mock
    ):
        response = _make_pii_response(
            redacted_files=[
                PiiFileResult(
                    file_name="other.pdf",
                    file_url="https://redacted/other.pdf",
                    pii_entities=[],
                )
            ],
        )
        client = _make_client(response)
        files = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
            )
        ]
        # When name doesn't match, masked_attachment_url falls back to the
        # original URL, so the upload pulls from there.
        httpx_mock.add_response(url="https://orig/doc.pdf", content=b"original-bytes")

        _, masked_files = await PiiMasker(client, None).apply("original", files)

        # url and name stay as the originals; masked_attachment_url is the
        # fallback (original URL).
        assert masked_files[0].url == "https://orig/doc.pdf"
        assert masked_files[0].name == "doc.pdf"
        assert masked_files[0].masked_attachment_url == "https://orig/doc.pdf"

    async def test_passes_entity_thresholds_from_policy(self):
        response = _make_pii_response()
        client = _make_client(response)
        policy = {
            "data": {
                "pii-entity-table": [
                    {
                        "pii-entity-is-enabled": True,
                        "pii-entity-category": "Email",
                        "pii-entity-confidence-threshold": 0.6,
                    }
                ]
            }
        }

        await PiiMasker(client, policy).apply("original", [])

        request = client.semantic_proxy.detect_pii_async.call_args[0][0]
        assert request.entity_thresholds == [
            PiiEntityThreshold(category="Email", confidence_threshold=0.6)
        ]


class TestRehydrate:
    """Test cases for PiiMasker.rehydrate."""

    def test_returns_text_unchanged_when_apply_not_called(self):
        masker = PiiMasker(Mock(), None)

        assert masker.rehydrate("nothing to rehydrate") == "nothing to rehydrate"

    async def test_rehydrates_placeholders_after_apply(self):
        entity = PiiEntity(
            pii_text="john@example.com",
            replacement_text="[EMAIL]",
            pii_type="Email",
            offset=8,
            confidence_score=0.95,
        )
        response = _make_pii_response(
            masked_prompt="contact [EMAIL]",
            entities=[entity],
            redacted_files=[
                PiiFileResult(
                    file_name="doc.pdf",
                    file_url="https://redacted/doc.pdf",
                    pii_entities=[entity],
                )
            ],
        )
        client = _make_client(response)

        masker = PiiMasker(client, None)
        await masker.apply("contact john@example.com", [])

        assert masker.rehydrate("Sent to [EMAIL]") == "Sent to john@example.com"
