"""Tests for projecting a run's attachments into guardrail attachment references."""

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import BuiltInValidatorGuardrail
from uipath.platform.guardrails.guardrails import EnumParameterValue

from uipath_langchain.agent.guardrails.attachment_refs import (
    _MAX_ATTACHMENTS,
    resolve_guardrail_attachments,
    resolve_referenced_attachments,
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


def _other_attachment(name: str = "b.pdf", mime: str = "application/pdf"):
    attachment_id = str(uuid.uuid4())
    return attachment_id, Attachment(
        id=uuid.UUID(attachment_id), full_name=name, mime_type=mime
    )


class TestResolveReferencedAttachments:
    """Tool scope: only the attachments a tool call mentions are forwarded."""

    async def test_resolves_the_attachment_the_tool_call_names(self):
        """The model passes ``{"ID": ...}`` only; name and type come from the registry."""
        result = resolve_referenced_attachments(
            {"attachment": {"ID": _UUID}, "question": "summarize"},
            _registry(),
            _judge(),
        )

        assert [r.model_dump(by_alias=True) for r in result] == [
            {"id": _UUID, "fileName": "a.csv", "mimeType": "text/csv"}
        ]

    async def test_forwards_nothing_when_the_call_names_no_file(self):
        """The run holds a file, but this call does not touch it: forwarding it would
        ship every file on every tool call."""
        result = resolve_referenced_attachments(
            {"query": "cats", "limit": 5}, _registry(), _judge()
        )

        assert result == []

    async def test_finds_mentions_nested_in_lists_and_dicts(self):
        other_id, other = _other_attachment()
        registry = _registry() | {other_id: other}
        data = {
            "files": [{"ID": _UUID}],
            "options": {"inner": {"ref": {"ID": other_id}, "flag": True}},
        }

        result = resolve_referenced_attachments(data, registry, _judge())

        assert sorted(r.id for r in result) == sorted([_UUID, other_id])

    async def test_skips_an_id_the_run_never_held_even_with_inline_name_and_type(
        self,
    ):
        """A tool result (or a prompt-injected argument) can name any id; only files the
        run legitimately holds are sent to the backend, which would otherwise look the
        id up in Orchestrator and judge a file the run never had."""
        other_id = str(uuid.uuid4())
        data = {"file": {"ID": other_id, "FullName": "out.csv", "MimeType": "text/csv"}}

        assert resolve_referenced_attachments(data, _registry(), _judge()) == []

    async def test_uses_the_registry_name_and_type_not_the_inline_ones(self):
        data = {"file": {"ID": _UUID, "FullName": "renamed.csv", "MimeType": "x/y"}}

        result = resolve_referenced_attachments(data, _registry(), _judge())

        assert [(r.file_name, r.mime_type) for r in result] == [("a.csv", "text/csv")]

    async def test_skips_a_uuid_that_is_not_one_of_the_run_attachments(self):
        """A UUID under ``ID`` that is not an attachment (a queue item, a job) must not
        reach the backend, which would look it up in Orchestrator."""
        data = {"item": {"ID": str(uuid.uuid4())}}

        assert resolve_referenced_attachments(data, _registry(), _judge()) == []

    @pytest.mark.parametrize("raw_id", ["queue-item-42", 12, True, None, ""])
    async def test_skips_a_non_uuid_id(self, raw_id):
        data = {"item": {"ID": raw_id, "FullName": "a.csv", "MimeType": "text/csv"}}

        assert resolve_referenced_attachments(data, _registry(), _judge()) == []

    async def test_deduplicates_the_same_attachment(self):
        data = {"a": {"ID": _UUID}, "b": [{"ID": _UUID.upper()}]}

        result = resolve_referenced_attachments(data, _registry(), _judge())

        assert [r.id for r in result] == [_UUID]

    async def test_caps_at_the_api_limit(self):
        registry = dict(_other_attachment() for _ in range(_MAX_ATTACHMENTS + 2))
        data = {"files": [{"ID": attachment_id} for attachment_id in registry]}

        result = resolve_referenced_attachments(data, registry, _judge())

        assert len(result) == _MAX_ATTACHMENTS

    @pytest.mark.parametrize("applies_to", ["Prompts", "prompts"])
    async def test_returns_empty_when_scoped_to_prompts(self, applies_to):
        result = resolve_referenced_attachments(
            {"attachment": {"ID": _UUID}}, _registry(), _scoped_judge(applies_to)
        )

        assert result == []

    async def test_accepts_attachment_instances_and_models(self):
        """Arguments may already carry expanded objects, not only wire dicts; they are
        still looked up in the registry by id."""

        class ToolArgs(BaseModel):
            attachment: Attachment

        registry = _registry()
        data = {
            "direct": registry[_UUID],
            "wrapped": ToolArgs(attachment=registry[_UUID]),
        }

        result = resolve_referenced_attachments(data, registry, _judge())

        assert [r.id for r in result] == [_UUID]

    async def test_bounds_the_scan_on_deep_and_large_payloads(self):
        """A tool result can be arbitrarily large; the scan stops instead of stalling the
        guardrail node, and never raises."""
        deep: dict[str, Any] = {"ID": _UUID}
        for _ in range(100):
            deep = {"child": deep}
        wide = {"rows": [{"n": i} for i in range(20_000)], "file": {"ID": _UUID}}

        assert resolve_referenced_attachments(deep, _registry(), _judge()) == []
        wide_result = resolve_referenced_attachments(wide, _registry(), _judge())
        assert isinstance(wide_result, list)

    @pytest.mark.parametrize("data", [None, 42, "just text", object(), [1, "two"]])
    async def test_never_raises_on_non_structured_payloads(self, data):
        assert resolve_referenced_attachments(data, _registry(), _judge()) == []

    async def test_matches_the_registry_path_for_the_same_attachment(self):
        """Agent/LLM scope and tool scope must send the backend identical references."""
        registry = _registry()

        via_registry = await resolve_guardrail_attachments(registry, _judge())
        via_mention = resolve_referenced_attachments(
            {"attachment": {"ID": _UUID}}, registry, _judge()
        )

        assert via_registry == via_mention


class _UnscannablePayload(dict[str, Any]):
    """A mapping whose lookups blow up, as a broken tool result might."""

    def get(self, key, default=None):  # noqa: D401
        raise RuntimeError("boom")


class _MentionWithUnreadableId(dict[str, Any]):
    """Looks like a mention to the scanner but fails when the id is read."""

    def get(self, key, default=None):
        return _UUID if key == "ID" else default

    def __getitem__(self, key):
        raise RuntimeError("boom")


class TestResolveReferencedAttachmentsErrorPaths:
    async def test_returns_empty_when_the_payload_cannot_be_scanned(self):
        """The guardrail node re-raises, so a broken payload must degrade to no files."""
        data = {"result": _UnscannablePayload(ID=_UUID)}

        assert resolve_referenced_attachments(data, _registry(), _judge()) == []

    async def test_skips_a_mention_whose_id_cannot_be_read(self):
        data = {"file": _MentionWithUnreadableId(), "other": {"ID": _UUID}}

        result = resolve_referenced_attachments(data, _registry(), _judge())

        assert [r.id for r in result] == [_UUID]
