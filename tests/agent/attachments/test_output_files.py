"""Tests for output-schema file field discovery, prompting, and verification."""

from typing import Any

import pytest

from uipath_langchain.agent.attachments.output_files import (
    build_output_files_prompt,
    get_output_file_fields,
    missing_output_files,
    output_attachment_ids,
    unlinked_output_attachment_ids,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    JOB_ATTACHMENT_DEFINITION,
)

ATTACHMENT_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ATTACHMENT_ID = "22222222-2222-2222-2222-222222222222"


def build_output_model(properties: dict[str, Any], required: list[str] | None = None):
    return create_model(
        {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
        }
    )


def ticket(attachment_id: str = ATTACHMENT_ID) -> dict[str, str]:
    return {
        "ID": attachment_id,
        "FullName": "report.md",
        "MimeType": "text/markdown",
    }


class TestGetOutputFileFields:
    def test_no_attachment_fields_returns_empty(self):
        model = build_output_model({"summary": {"type": "string"}})
        assert get_output_file_fields(model) == []

    def test_discovers_name_description_and_required(self):
        model = build_output_model(
            {
                "summary": {"type": "string"},
                "report": {
                    "$ref": "#/definitions/job-attachment",
                    "description": "The generated report",
                },
            },
            required=["summary", "report"],
        )

        fields = get_output_file_fields(model)

        assert len(fields) == 1
        assert fields[0].path == "$.report"
        assert fields[0].name == "report"
        assert fields[0].description == "The generated report"
        assert fields[0].required is True

    def test_optional_field_is_not_required(self):
        model = build_output_model(
            {"report": {"$ref": "#/definitions/job-attachment"}},
        )

        assert get_output_file_fields(model)[0].required is False

    def test_array_of_attachments_keeps_the_field_name(self):
        model = build_output_model(
            {
                "exports": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                    "description": "Every exported file",
                }
            },
            required=["exports"],
        )

        field = get_output_file_fields(model)[0]

        assert field.path == "$.exports[*]"
        assert field.name == "exports"
        assert field.description == "Every exported file"


class TestBuildOutputFilesPrompt:
    def test_empty_fields_produce_no_prompt(self):
        assert build_output_files_prompt([], tool_name="create_output_file") == ""

    def test_lists_each_field_with_its_description(self):
        model = build_output_model(
            {
                "report": {
                    "$ref": "#/definitions/job-attachment",
                    "description": "The generated report",
                },
                "extras": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                },
            },
            required=["report"],
        )

        prompt = build_output_files_prompt(
            get_output_file_fields(model), tool_name="create_output_file"
        )

        assert "create_output_file" in prompt
        assert "`report` (required) — The generated report" in prompt
        assert "`extras` (optional)" in prompt
        assert "choose the format that best fits the content" in prompt

    def test_required_field_is_told_to_produce_the_file(self):
        model = build_output_model(
            {"report": {"$ref": "#/definitions/job-attachment"}}, required=["report"]
        )

        prompt = build_output_files_prompt(
            get_output_file_fields(model), tool_name="create_output_file"
        )

        assert "Create every required file" in prompt
        assert "only when it serves the request" not in prompt

    def test_optional_field_is_not_told_to_produce_the_file(self):
        """The runtime does not require it, so the prompt must not demand it."""
        model = build_output_model({"report": {"$ref": "#/definitions/job-attachment"}})

        prompt = build_output_files_prompt(
            get_output_file_fields(model), tool_name="create_output_file"
        )

        assert "only when it serves the request" in prompt
        assert "Create every required file" not in prompt

    def test_mixed_fields_state_both_rules(self):
        model = build_output_model(
            {
                "report": {"$ref": "#/definitions/job-attachment"},
                "extras": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                },
            },
            required=["report"],
        )

        prompt = build_output_files_prompt(
            get_output_file_fields(model), tool_name="create_output_file"
        )

        assert "Create every required file" in prompt
        assert "only when it serves the request" in prompt

    def test_workspace_rule_only_when_requested(self):
        model = build_output_model({"report": {"$ref": "#/definitions/job-attachment"}})
        fields = get_output_file_fields(model)

        assert "file_path" not in build_output_files_prompt(
            fields, tool_name="create_output_file"
        )
        assert "file_path" in build_output_files_prompt(
            fields, tool_name="create_output_file", with_workspace=True
        )


class TestMissingOutputFiles:
    @pytest.fixture
    def fields(self):
        model = build_output_model(
            {
                "report": {"$ref": "#/definitions/job-attachment"},
                "optional_export": {"$ref": "#/definitions/job-attachment"},
            },
            required=["report"],
        )
        return get_output_file_fields(model)

    def test_required_field_absent_is_reported(self, fields):
        missing = missing_output_files(fields, {"summary": "done"})

        assert [field.name for field in missing] == ["report"]

    def test_required_field_null_is_reported(self, fields):
        missing = missing_output_files(fields, {"report": None})

        assert [field.name for field in missing] == ["report"]

    def test_required_field_filled_is_not_reported(self, fields):
        assert missing_output_files(fields, {"report": ticket()}) == []

    def test_optional_field_absent_is_not_reported(self, fields):
        assert missing_output_files(fields, {"report": ticket()}) == []


class TestOutputAttachmentIds:
    @pytest.fixture
    def fields(self):
        model = build_output_model(
            {
                "report": {"$ref": "#/definitions/job-attachment"},
                "exports": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                },
            }
        )
        return get_output_file_fields(model)

    def test_collects_ids_from_scalar_and_array_fields(self, fields):
        ids = output_attachment_ids(
            fields,
            {"report": ticket(), "exports": [ticket(OTHER_ATTACHMENT_ID)]},
        )

        assert sorted(ids) == sorted([ATTACHMENT_ID, OTHER_ATTACHMENT_ID])

    def test_ignores_empty_and_malformed_values(self, fields):
        ids = output_attachment_ids(
            fields, {"report": None, "exports": [{"FullName": "x.md"}]}
        )

        assert ids == []


class TestUnlinkedOutputAttachmentIds:
    @pytest.fixture
    def fields(self):
        model = build_output_model(
            {"report": {"$ref": "#/definitions/job-attachment"}}, required=["report"]
        )
        return get_output_file_fields(model)

    async def test_no_job_key_skips_verification(self, fields, monkeypatch):
        monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)

        assert await unlinked_output_attachment_ids(fields, {"report": ticket()}) == []

    async def test_linked_attachment_passes(self, fields, monkeypatch):
        _patch_job(monkeypatch, linked=[ATTACHMENT_ID])

        assert await unlinked_output_attachment_ids(fields, {"report": ticket()}) == []

    async def test_linked_attachment_matches_case_insensitively(
        self, fields, monkeypatch
    ):
        _patch_job(monkeypatch, linked=[ATTACHMENT_ID.upper()])

        assert await unlinked_output_attachment_ids(fields, {"report": ticket()}) == []

    async def test_unknown_attachment_is_reported(self, fields, monkeypatch):
        _patch_job(monkeypatch, linked=[OTHER_ATTACHMENT_ID])

        unlinked = await unlinked_output_attachment_ids(fields, {"report": ticket()})

        assert unlinked == [ATTACHMENT_ID]

    async def test_empty_output_does_not_call_the_platform(self, fields, monkeypatch):
        calls: list[Any] = []
        _patch_job(monkeypatch, linked=[], calls=calls)

        assert await unlinked_output_attachment_ids(fields, {}) == []
        assert calls == []


def _patch_job(
    monkeypatch, *, linked: list[str], calls: list[Any] | None = None
) -> None:
    """Point the verification at a fake job with ``linked`` attachments."""
    monkeypatch.setenv("UIPATH_JOB_KEY", "33333333-3333-3333-3333-333333333333")
    monkeypatch.delenv("UIPATH_FOLDER_KEY", raising=False)

    class FakeJobs:
        async def list_attachments_async(self, **kwargs: Any) -> list[str]:
            if calls is not None:
                calls.append(kwargs)
            return linked

    class FakeUiPath:
        jobs = FakeJobs()

    monkeypatch.setattr(
        "uipath_langchain.agent.attachments.output_files.UiPath",
        lambda *args, **kwargs: FakeUiPath(),
    )
