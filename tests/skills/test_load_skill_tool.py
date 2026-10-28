"""Tests for the ``load_skill`` tool."""

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.platform.skills import SkillSet

from uipath_langchain.skills import create_load_skill_tool


def test_tool_lists_qualified_names_as_enum(skill_set: SkillSet) -> None:
    tool = create_load_skill_tool(skill_set)
    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert schema["properties"]["name"]["enum"] == [
        "finance/invoice-triage",
        "finance/vendor-lookup",
        "ops/vendor-lookup",
    ]
    assert schema["properties"]["path"]["default"] == "SKILL.md"
    assert schema["required"] == ["name"]


def test_tool_returns_header_body_and_files(skill_set: SkillSet) -> None:
    out = create_load_skill_tool(skill_set).invoke({"name": "finance/invoice-triage"})
    assert out.startswith("# finance/invoice-triage (bundle finance v3)")
    assert "# Invoice triage" in out
    assert out.endswith("Supporting files: reference/vendors.md")


def test_tool_without_supporting_files_has_no_footer(skill_set: SkillSet) -> None:
    out = create_load_skill_tool(skill_set).invoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup (bundle ops v1)")
    assert "Supporting files:" not in out


def test_tool_reads_supporting_file(skill_set: SkillSet) -> None:
    out = create_load_skill_tool(skill_set).invoke(
        {"name": "finance/invoice-triage", "path": "reference/vendors.md"}
    )
    assert out.startswith("# finance/invoice-triage — reference/vendors.md")
    assert "# Vendor normalization" in out


def test_tool_unknown_name_returns_message_not_error(skill_set: SkillSet) -> None:
    # Calling ``func`` bypasses the enum on purpose.
    out = create_load_skill_tool(skill_set).func(name="finance/nope")  # type: ignore[misc]
    assert out == (
        "Unknown skill 'finance/nope'. Available: finance/invoice-triage, "
        "finance/vendor-lookup, ops/vendor-lookup"
    )


def test_tool_unknown_path_lists_files(skill_set: SkillSet) -> None:
    out = create_load_skill_tool(skill_set).func(  # type: ignore[misc]
        name="finance/invoice-triage", path="nope.md"
    )
    assert out == (
        "Skill 'finance/invoice-triage' has no file 'nope.md'. "
        "Files: SKILL.md, reference/vendors.md"
    )


def test_tool_cleaned_up_set_reports_unavailable(skill_set: SkillSet) -> None:
    tool = create_load_skill_tool(skill_set)
    skill_set.root = None
    out = tool.func(name="finance/invoice-triage")  # type: ignore[misc]
    assert out.startswith("Skill 'finance/invoice-triage' is not available any more: ")


async def test_tool_async(skill_set: SkillSet) -> None:
    out = await create_load_skill_tool(skill_set).ainvoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup (bundle ops v1)")


def test_tool_metadata_and_name(skill_set: SkillSet) -> None:
    tool = create_load_skill_tool(skill_set)
    assert tool.name == "load_skill"
    assert tool.metadata == {"tool_type": "internal", "display_name": "load_skill"}
    assert "## Skills" in tool.description
    renamed = create_load_skill_tool(skill_set, name="read_skill")
    assert renamed.name == "read_skill"
    assert renamed.metadata == {"tool_type": "internal", "display_name": "read_skill"}


def test_tool_with_empty_set_accepts_any_name_and_reports_none(
    empty_skill_set: SkillSet,
) -> None:
    tool = create_load_skill_tool(empty_skill_set)
    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert "enum" not in schema["properties"]["name"]
    assert schema["properties"]["name"]["type"] == "string"
    assert tool.func(name="x") == "Unknown skill 'x'. Available: none"  # type: ignore[misc]


def test_tool_records_one_skill_load_span(
    skill_set: SkillSet, exporter: InMemorySpanExporter
) -> None:
    tool = create_load_skill_tool(skill_set)
    exporter.clear()
    tool.invoke({"name": "finance/invoice-triage"})
    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["Skill load - finance/invoice-triage"]
    attributes = dict(spans[0].attributes or {})
    assert attributes["skill.invoked_by"] == "model"
    assert attributes["skill.bundle.name"] == "finance"
    assert attributes["skill.version"] == "3"


def test_tool_supporting_file_records_no_span(
    skill_set: SkillSet, exporter: InMemorySpanExporter
) -> None:
    tool = create_load_skill_tool(skill_set)
    exporter.clear()
    tool.invoke({"name": "finance/invoice-triage", "path": "reference/vendors.md"})
    assert exporter.get_finished_spans() == ()
