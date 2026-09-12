"""Tests for projecting the job-attachment registry into guardrail attachment refs."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from uipath.platform.attachments import Attachment, BlobFileAccessInfo
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from uipath_langchain.agent.guardrails.attachment_refs import (
    GUARDRAIL_ATTACHMENTS_FEATURE_FLAG,
    resolve_guardrail_attachments,
)

_UUID = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"
_ENV_FLAG = f"UIPATH_FEATURE_{GUARDRAIL_ATTACHMENTS_FEATURE_FLAG}"


def _judge() -> MagicMock:
    guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
    guardrail.validator_type = "llm_as_judge"
    return guardrail


def _registry(mime: str = "text/csv", name: str = "a.csv") -> dict[str, Attachment]:
    return {_UUID: Attachment(ID=_UUID, FullName=name, MimeType=mime)}


def _patch_client(monkeypatch, *, uri: str = "", name: str = "", raises=None):
    """Patch the UiPath client used by the resolver."""
    attachments = MagicMock()
    if raises is not None:
        attachments.get_blob_file_access_uri_async = AsyncMock(side_effect=raises)
    else:
        attachments.get_blob_file_access_uri_async = AsyncMock(
            return_value=BlobFileAccessInfo(id=uuid.UUID(_UUID), uri=uri, name=name)
        )
    client = MagicMock()
    client.attachments = attachments
    monkeypatch.setattr(
        "uipath_langchain.agent.guardrails.attachment_refs.UiPath",
        lambda: client,
    )
    return client


class TestResolveGuardrailAttachments:
    async def test_resolves_text_attachment(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "true")
        _patch_client(monkeypatch, uri="https://x/a.csv?sig=s", name="a.csv")

        result = await resolve_guardrail_attachments(_registry(), _judge())

        assert [r.model_dump(by_alias=True) for r in result] == [
            {
                "id": _UUID,
                "fileName": "a.csv",
                "mimeType": "text/csv",
                "url": "https://x/a.csv?sig=s",
            }
        ]

    async def test_returns_empty_when_flag_off(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "false")
        client = _patch_client(monkeypatch, uri="https://x/a.csv", name="a.csv")

        assert await resolve_guardrail_attachments(_registry(), _judge()) == []
        client.attachments.get_blob_file_access_uri_async.assert_not_awaited()

    async def test_returns_empty_for_non_judge_validator(self, monkeypatch):
        """Resolving a SAS url costs an Orchestrator call; don't spend it for nothing."""
        monkeypatch.setenv(_ENV_FLAG, "true")
        client = _patch_client(monkeypatch, uri="https://x/a.csv", name="a.csv")
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.validator_type = "pii_detection"

        assert await resolve_guardrail_attachments(_registry(), guardrail) == []
        client.attachments.get_blob_file_access_uri_async.assert_not_awaited()

    @pytest.mark.parametrize(
        "mime", ["application/octet-stream", "application/zip", "video/mp4"]
    )
    async def test_skips_unsupported_mime_type(self, monkeypatch, mime):
        monkeypatch.setenv(_ENV_FLAG, "true")
        _patch_client(monkeypatch, uri="https://x/a", name="a")

        assert await resolve_guardrail_attachments(_registry(mime=mime), _judge()) == []

    @pytest.mark.parametrize(
        "mime",
        [
            # Phase 1 — decoded and inlined by the backend.
            "text/plain",
            "text/csv",
            "application/json",
            "text/markdown",
            # Phase 2 — sent to a vision-capable judge as content parts.
            "application/pdf",
            "image/png",
            "image/jpeg",
        ],
    )
    async def test_accepts_every_supported_mime_type(self, monkeypatch, mime):
        monkeypatch.setenv(_ENV_FLAG, "true")
        _patch_client(monkeypatch, uri="https://x/a", name="a")

        result = await resolve_guardrail_attachments(_registry(mime=mime), _judge())

        assert len(result) == 1

    async def test_never_raises_when_resolution_fails(self, monkeypatch):
        """The low-code guardrail node re-raises everything, so this must absorb it.

        A transient Orchestrator failure killing a production run is a far worse
        outcome than an unscanned file.
        """
        monkeypatch.setenv(_ENV_FLAG, "true")
        _patch_client(monkeypatch, raises=RuntimeError("orchestrator 503"))

        assert await resolve_guardrail_attachments(_registry(), _judge()) == []

    async def test_partial_failure_keeps_the_readable_attachments(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "true")
        good_id, bad_id = str(uuid.uuid4()), str(uuid.uuid4())
        registry = {
            good_id: Attachment(ID=good_id, FullName="good.csv", MimeType="text/csv"),
            bad_id: Attachment(ID=bad_id, FullName="bad.csv", MimeType="text/csv"),
        }

        async def _resolve(*, key):
            if str(key) == bad_id:
                raise RuntimeError("gone")
            return BlobFileAccessInfo(id=key, uri="https://x/good.csv", name="good.csv")

        attachments = MagicMock()
        attachments.get_blob_file_access_uri_async = AsyncMock(side_effect=_resolve)
        client = MagicMock()
        client.attachments = attachments
        monkeypatch.setattr(
            "uipath_langchain.agent.guardrails.attachment_refs.UiPath", lambda: client
        )

        result = await resolve_guardrail_attachments(registry, _judge())

        assert [r.file_name for r in result] == ["good.csv"]

    async def test_caps_attachment_count(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "true")
        _patch_client(monkeypatch, uri="https://x/a.csv", name="a.csv")
        registry = {}
        for index in range(10):
            attachment_id = str(uuid.uuid4())
            registry[attachment_id] = Attachment(
                ID=attachment_id, FullName=f"{index}.csv", MimeType="text/csv"
            )

        result = await resolve_guardrail_attachments(registry, _judge())

        assert len(result) == 5

    async def test_returns_empty_for_empty_registry(self, monkeypatch):
        monkeypatch.setenv(_ENV_FLAG, "true")
        _patch_client(monkeypatch, uri="https://x/a.csv", name="a.csv")

        assert await resolve_guardrail_attachments({}, _judge()) == []
