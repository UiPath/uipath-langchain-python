"""Tests for output-schema file field discovery and verification."""

from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from uipath_langchain.agent.attachments.output_files import (
    build_files_prompt,
    check_output_files,
    get_output_file_fields,
    has_attachment_fields,
    malformed_output_files,
    missing_output_files,
    output_attachment_ids,
    resolve_output_attachments,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    JOB_ATTACHMENT_DEFINITION,
)

from .fake_orchestrator import patch_orchestrator

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


class TestAliasedFileFields:
    """A property whose name collides with a BaseModel attribute is aliased by
    the converter, so matching on model_fields keys alone would miss it."""

    @pytest.mark.parametrize("json_name", ["schema", "copy", "json", "dict"])
    def test_required_aliased_field_keeps_its_metadata(self, json_name):
        model = build_output_model(
            {
                json_name: {
                    "$ref": "#/definitions/job-attachment",
                    "description": "The generated report",
                }
            },
            required=[json_name],
        )

        field = get_output_file_fields(model)[0]

        assert field.name == json_name
        assert field.required is True
        assert field.description == "The generated report"

    def test_required_aliased_field_is_flagged_when_empty(self):
        """Without this the retry gate never fires and termination hard-fails."""
        model = build_output_model(
            {"schema": {"$ref": "#/definitions/job-attachment"}}, required=["schema"]
        )
        fields = get_output_file_fields(model)

        assert [f.name for f in missing_output_files(fields, {})] == ["schema"]


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

    async def test_diagnosis_prefers_an_existing_reference_and_names_the_tool(
        self, fields
    ):
        problem = (
            await check_output_files(fields, {}, file_tool_name="Create_File")
        ).problem

        assert problem is not None
        assert "returned by a tool" in problem
        assert "`Create_File`" in problem

    async def test_diagnosis_without_the_tool_does_not_name_one(self, fields):
        problem = (await check_output_files(fields, {})).problem

        assert problem is not None
        assert "a tool that creates files" in problem

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


class TestResolveOutputAttachments:
    @pytest.fixture
    def fields(self):
        model = build_output_model(
            {"report": {"$ref": "#/definitions/job-attachment"}}, required=["report"]
        )
        return get_output_file_fields(model)

    async def test_no_job_key_passes_the_output_through(self, fields, monkeypatch):
        monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
        output = {"report": ticket()}

        assert await resolve_output_attachments(fields, output) == (output, [])

    async def test_an_attachment_from_another_job_is_accepted_and_linked(
        self, fields, monkeypatch
    ):
        fake = patch_orchestrator(
            monkeypatch, existing={ATTACHMENT_ID: "report.md"}, linked=[]
        )

        output, unknown = await resolve_output_attachments(fields, {"report": ticket()})

        assert unknown == []
        assert output["report"]["ID"] == ATTACHMENT_ID
        assert fake.links == [ATTACHMENT_ID]

    async def test_an_already_linked_attachment_is_not_linked_again(
        self, fields, monkeypatch
    ):
        fake = patch_orchestrator(
            monkeypatch,
            existing={ATTACHMENT_ID: "report.md"},
            linked=[ATTACHMENT_ID.upper()],
        )

        await resolve_output_attachments(fields, {"report": ticket()})

        assert fake.links == []

    async def test_the_reference_is_rebuilt_from_the_attachment(
        self, fields, monkeypatch
    ):
        patch_orchestrator(monkeypatch, existing={ATTACHMENT_ID: "accounts.csv"})
        edited = {"ID": ATTACHMENT_ID, "FullName": "/x.md", "MimeType": "text/plain"}

        output, _ = await resolve_output_attachments(fields, {"report": edited})

        assert output["report"] == {
            "ID": ATTACHMENT_ID,
            "FullName": "accounts.csv",
            "MimeType": "text/csv",
        }

    async def test_an_unknown_attachment_is_reported_and_nothing_is_linked(
        self, fields, monkeypatch
    ):
        fake = patch_orchestrator(monkeypatch, existing={})

        output, unknown = await resolve_output_attachments(fields, {"report": ticket()})

        assert unknown == [ATTACHMENT_ID]
        assert fake.links == []

    async def test_empty_output_does_not_call_the_platform(self, fields, monkeypatch):
        fake = patch_orchestrator(monkeypatch, existing={})

        assert await resolve_output_attachments(fields, {}) == ({}, [])
        assert fake.lookups == []


class TestMalformedOutputFiles:
    """A half-filled reference must be corrected, not faulted on.

    Anything the output schema would reject has to be caught here: past the
    gate, termination validates and raises, so the agent never gets its turn.
    """

    @pytest.fixture
    def fields(self):
        model = build_output_model(
            {"report": {"$ref": "#/definitions/job-attachment"}}, required=["report"]
        )
        return get_output_file_fields(model)

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("no id", {"FullName": "x.txt", "MimeType": "text/plain"}),
            ("empty id", {"ID": "", "FullName": "x.txt", "MimeType": "text/plain"}),
            ("id only", {"ID": ATTACHMENT_ID}),
            ("no mime type", {"ID": ATTACHMENT_ID, "FullName": "x.txt"}),
            (
                "id not a uuid",
                {"ID": "nope", "FullName": "x", "MimeType": "text/plain"},
            ),
        ],
    )
    def test_unusable_reference_is_reported(self, fields, label, value):
        assert [f.name for f in malformed_output_files(fields, {"report": value})] == [
            "report"
        ]

    def test_complete_reference_is_accepted(self, fields):
        assert malformed_output_files(fields, {"report": ticket()}) == []

    def test_empty_field_is_left_to_the_missing_check(self, fields):
        """Empty is a different problem with a different message."""
        assert malformed_output_files(fields, {"report": None}) == []
        assert [f.name for f in missing_output_files(fields, {"report": None})] == [
            "report"
        ]

    async def test_diagnosis_names_the_field_without_calling_the_platform(
        self, fields, monkeypatch
    ):
        fake = patch_orchestrator(monkeypatch, existing={})

        problem = (
            await check_output_files(fields, {"report": {"FullName": "x.txt"}})
        ).problem

        assert problem is not None
        assert "'report'" in problem
        # A shape problem is settled locally; no point asking Orchestrator.
        assert fake.lookups == []


class TestMessagesNameTheTool:
    async def test_malformed_reference_names_the_tool(self, monkeypatch):
        fields = get_output_file_fields(
            build_output_model({"report": {"$ref": "#/definitions/job-attachment"}})
        )
        patch_orchestrator(monkeypatch, existing={})

        problem = (
            await check_output_files(
                fields, {"report": {"FullName": "x.md"}}, file_tool_name="Create_File"
            )
        ).problem

        assert problem is not None
        assert "`Create_File`" in problem

    async def test_unknown_reference_names_the_tool(self, monkeypatch):
        fields = get_output_file_fields(
            build_output_model({"report": {"$ref": "#/definitions/job-attachment"}})
        )
        patch_orchestrator(monkeypatch, existing={})

        problem = (
            await check_output_files(
                fields, {"report": ticket()}, file_tool_name="Create_File"
            )
        ).problem

        assert problem is not None
        assert "do not name an existing file" in problem
        assert "`Create_File`" in problem


class TestHasAttachmentFields:
    def test_output_file_field(self):
        model = build_output_model({"report": {"$ref": "#/definitions/job-attachment"}})

        assert has_attachment_fields([], model)

    def test_tool_argument_file_field(self):
        tool = StructuredTool(
            name="t",
            description="d",
            args_schema=build_output_model(
                {"document": {"$ref": "#/definitions/job-attachment"}}
            ),
            func=lambda **_: None,
        )

        assert has_attachment_fields([tool], None)

    def test_no_file_field(self):
        model = build_output_model({"summary": {"type": "string"}})

        assert not has_attachment_fields([], model)


class TestBuildFilesPrompt:
    def test_standard_never_mentions_a_workspace(self):
        prompt = build_files_prompt(file_tool_name="Create_File", with_workspace=False)

        assert "workspace" not in prompt
        assert "`Create_File`" in prompt

    def test_advanced_explains_workspace_files(self):
        prompt = build_files_prompt(file_tool_name="Create_File", with_workspace=True)

        assert "Files in your workspace are not attachments" in prompt

    def test_without_the_tool_no_tool_is_named(self):
        prompt = build_files_prompt(file_tool_name=None, with_workspace=False)

        assert "create it with" not in prompt
        assert "Never write a reference yourself." in prompt
