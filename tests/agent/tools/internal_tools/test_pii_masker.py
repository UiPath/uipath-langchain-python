"""Tests for pii_masker.py module."""

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


def _make_client(response: PiiDetectionResponse) -> Mock:
    client = Mock()
    client.semantic_proxy = Mock()
    client.semantic_proxy.detect_pii_async = AsyncMock(return_value=response)
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


class TestRenameForMasking:
    """Test cases for PiiMasker._rename_for_masking."""

    def test_file_with_extension_preserves_extension(self):
        original = FileInfo(
            url="https://orig/report.pdf",
            name="report.pdf",
            mime_type="application/pdf",
        )

        renamed = PiiMasker._rename_for_masking(original, "https://redacted/report.pdf")

        assert renamed.url == "https://redacted/report.pdf"
        assert renamed.name == "pii_masked_report.pdf"
        assert renamed.mime_type == "application/pdf"

    def test_file_without_extension_uses_whole_name(self):
        original = FileInfo(
            url="https://orig/readme", name="readme", mime_type="text/plain"
        )

        renamed = PiiMasker._rename_for_masking(original, "https://redacted/readme")

        assert renamed.name == "pii_masked_readme"
        assert renamed.url == "https://redacted/readme"
        assert renamed.mime_type == "text/plain"

    def test_file_with_multiple_dots_splits_at_last(self):
        original = FileInfo(
            url="https://orig/data.backup.json",
            name="data.backup.json",
            mime_type="application/json",
        )

        renamed = PiiMasker._rename_for_masking(
            original, "https://redacted/data.backup.json"
        )

        assert renamed.name == "pii_masked_data.backup.json"


class TestApply:
    """Test cases for PiiMasker.apply."""

    async def test_masks_prompt_and_renames_files(self):
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
        client = _make_client(response)
        files = [
            FileInfo(
                url="https://orig/doc.pdf",
                name="doc.pdf",
                mime_type="application/pdf",
            )
        ]

        masker = PiiMasker(client, policy=None)
        masked_prompt, masked_files = await masker.apply(
            "Please contact john@example.com for info", files
        )

        assert masked_prompt == "Please contact [EMAIL] for info"
        assert len(masked_files) == 1
        assert masked_files[0].url == "https://redacted/doc.pdf"
        assert masked_files[0].name == "pii_masked_doc.pdf"

        request = client.semantic_proxy.detect_pii_async.call_args[0][0]
        assert request.files[0].file_name == "doc.pdf"
        assert request.files[0].file_type == "pdf"

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

    async def test_unmatched_redacted_filename_keeps_original_url(self):
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

        _, masked_files = await PiiMasker(client, None).apply("original", files)

        # Rename still applies; URL falls back to the original when unmatched.
        assert masked_files[0].url == "https://orig/doc.pdf"
        assert masked_files[0].name == "pii_masked_doc.pdf"

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
