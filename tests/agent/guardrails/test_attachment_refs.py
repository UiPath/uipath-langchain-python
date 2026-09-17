"""Tests for projecting the job-attachment registry into guardrail attachment refs."""

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import BuiltInValidatorGuardrail
from uipath.platform.guardrails.guardrails import EnumParameterValue

from uipath_langchain.agent.guardrails.attachment_refs import (
    _MAX_ATTACHMENTS,
    resolve_guardrail_attachments,
)

_UUID = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"


def _judge() -> MagicMock:
    guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
    guardrail.name = "Example"
    guardrail.validator_type = "llm_as_judge"
    return guardrail


def _scoped_judge(applies_to: str, parameter_id: str = "appliesTo") -> MagicMock:
    """A judge guardrail carrying the ``appliesTo`` parameter the designer writes."""
    guardrail = _judge()
    guardrail.validator_parameters = [
        EnumParameterValue.model_validate(
            {"$parameterType": "enum", "id": parameter_id, "value": applies_to}
        )
    ]
    return guardrail


def _registry(mime: str = "text/csv", name: str = "a.csv") -> dict[str, Attachment]:
    return {_UUID: Attachment(id=uuid.UUID(_UUID), full_name=name, mime_type=mime)}


class TestResolveGuardrailAttachments:
    async def test_resolves_text_attachment(self, monkeypatch):
        result = await resolve_guardrail_attachments(_registry(), _judge())

        assert [r.model_dump(by_alias=True) for r in result] == [
            {
                "id": _UUID,
                "fileName": "a.csv",
                "mimeType": "text/csv",
            }
        ]

    @pytest.mark.parametrize(
        "validator_type", ["pii_detection", "user_prompt_attacks", "harmful_content"]
    )
    async def test_resolves_for_any_validator(self, monkeypatch, validator_type):
        """The runtime forwards for every guardrail; the backend decides who can use it."""
        guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
        guardrail.validator_type = validator_type

        result = await resolve_guardrail_attachments(_registry(), guardrail)

        assert [r.file_name for r in result] == ["a.csv"]

    @pytest.mark.parametrize(
        "mime", ["application/octet-stream", "application/zip", "video/mp4"]
    )
    async def test_forwards_any_mime_type(self, monkeypatch, mime):
        """No type filter here: the backend skips (and logs) what it cannot inspect."""
        result = await resolve_guardrail_attachments(_registry(mime=mime), _judge())

        assert [r.mime_type for r in result] == [mime]

    @pytest.mark.parametrize("mime,name", [("", "a.csv"), ("text/csv", "")])
    async def test_skips_attachment_missing_name_or_type(self, monkeypatch, mime, name):
        """The validate API requires both; forwarding an empty one would 400 the call."""
        result = await resolve_guardrail_attachments(
            _registry(mime=mime, name=name), _judge()
        )

        assert result == []

    async def test_skips_attachment_with_non_uuid_id(self, monkeypatch):
        """The validate API requires a GUID; a malformed id must not reach it, and this
        module never raises over it either."""
        attachment = MagicMock(id="not-a-uuid", full_name="a.csv", mime_type="text/csv")

        result = await resolve_guardrail_attachments(
            {"not-a-uuid": attachment}, _judge()
        )

        assert result == []

    async def test_caps_attachment_count(self, monkeypatch):
        registry = {}
        for index in range(10):
            attachment_id = str(uuid.uuid4())
            registry[attachment_id] = Attachment(
                id=uuid.UUID(attachment_id),
                full_name=f"{index}.csv",
                mime_type="text/csv",
            )

        result = await resolve_guardrail_attachments(registry, _judge())

        assert len(result) == 5

    async def test_malformed_entry_neither_raises_nor_consumes_a_slot(
        self, monkeypatch
    ):
        """One bad registry value must not end the run or hide a later valid file."""
        registry: dict[str, Any] = {"bad": object()}
        for index in range(_MAX_ATTACHMENTS):
            attachment_id = str(uuid.uuid4())
            registry[attachment_id] = Attachment(
                id=uuid.UUID(attachment_id),
                full_name=f"{index}.csv",
                mime_type="text/csv",
            )

        result = await resolve_guardrail_attachments(registry, _judge())

        assert [a.file_name for a in result] == [
            f"{i}.csv" for i in range(_MAX_ATTACHMENTS)
        ]

    async def test_returns_empty_for_empty_registry(self, monkeypatch):
        assert await resolve_guardrail_attachments({}, _judge()) == []

    async def test_truncates_over_long_file_names_to_the_api_ceiling(self, monkeypatch):
        """The validate API rejects names over 260 chars; a 400 there would kill the run."""
        long_name = "x" * 300 + ".csv"

        result = await resolve_guardrail_attachments(
            _registry(name=long_name), _judge()
        )

        assert len(result[0].file_name) == 260

    @pytest.mark.parametrize("applies_to", ["Prompts", "prompts", "  PROMPTS  "])
    async def test_returns_empty_when_scoped_to_prompts(self, monkeypatch, applies_to):
        """A prompts-only guardrail must not forward any file reference."""
        result = await resolve_guardrail_attachments(
            _registry(), _scoped_judge(applies_to)
        )

        assert result == []

    async def test_matches_the_scope_parameter_id_case_insensitively(self, monkeypatch):
        """The backend matches parameter ids ignoring case; a mismatch here would resolve
        files the author scoped out."""
        result = await resolve_guardrail_attachments(
            _registry(), _scoped_judge("Prompts", parameter_id="AppliesTo")
        )

        assert result == []

    @pytest.mark.parametrize(
        "applies_to", ["Files", "Both", "both", "something-we-never-shipped"]
    )
    async def test_resolves_when_the_scope_is_not_prompts_only(
        self, monkeypatch, applies_to
    ):
        """Anything but Prompts keeps files in scope, matching the backend's default of Both.
        An unrecognized value must not silently stop scanning files."""
        result = await resolve_guardrail_attachments(
            _registry(), _scoped_judge(applies_to)
        )

        assert [r.file_name for r in result] == ["a.csv"]

    async def test_resolves_when_the_scope_parameter_is_malformed(self, monkeypatch):
        """Never raises: the caller re-raises, which would end the run over a bad parameter."""
        guardrail = _judge()
        guardrail.validator_parameters = 7  # not a list

        result = await resolve_guardrail_attachments(_registry(), guardrail)

        assert [r.file_name for r in result] == ["a.csv"]
