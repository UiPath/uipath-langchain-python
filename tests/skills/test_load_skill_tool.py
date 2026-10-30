"""Tests for the ``load_skill`` tool."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel
from uipath.platform.skills import SkillSet

from uipath_langchain.skills import create_load_skill_tool
from uipath_langchain.skills import load_skill_tool as module


def _schema(tool: StructuredTool) -> dict[str, Any]:
    args_schema = tool.args_schema
    assert isinstance(args_schema, type) and issubclass(args_schema, BaseModel)
    return args_schema.model_json_schema()


def _call(tool: StructuredTool, **kwargs: str) -> str:
    """Call the sync implementation directly, bypassing argument validation."""
    assert tool.func is not None
    return tool.func(**kwargs)


def _file(skill_set: SkillSet, qualified: str, path: str = "SKILL.md") -> Path:
    assert skill_set.root is not None
    return skill_set.root.joinpath(*qualified.split("/"), path)


def test_tool_lists_qualified_names_as_enum(skill_set: SkillSet) -> None:
    schema = _schema(create_load_skill_tool(skill_set))
    assert schema["properties"]["name"]["enum"] == [
        "finance/invoice-triage",
        "finance/vendor-lookup",
        "ops/vendor-lookup",
    ]
    assert schema["properties"]["path"]["default"] == "SKILL.md"
    assert schema["required"] == ["name"]


def test_tool_name_is_a_plain_string_above_the_enum_cap(
    skill_set: SkillSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_ENUM_NAMES", 1)
    tool = create_load_skill_tool(skill_set)
    name = _schema(tool)["properties"]["name"]
    assert "enum" not in name
    assert name["type"] == "string"
    assert tool.invoke({"name": "ops/vendor-lookup"}).startswith("# ops/vendor-lookup")


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


def test_tool_normalizes_a_dot_slash_path(skill_set: SkillSet) -> None:
    out = create_load_skill_tool(skill_set).invoke(
        {"name": "finance/invoice-triage", "path": "./reference/vendors.md"}
    )
    assert out.startswith("# finance/invoice-triage — reference/vendors.md")
    assert "# Vendor normalization" in out


def test_tool_unknown_name_returns_message_not_error(skill_set: SkillSet) -> None:
    out = _call(create_load_skill_tool(skill_set), name="finance/nope")
    assert out == (
        "Unknown skill 'finance/nope'. Available: finance/invoice-triage, "
        "finance/vendor-lookup, ops/vendor-lookup"
    )


def test_tool_node_turns_an_unknown_name_into_an_error_message(
    skill_set: SkillSet,
) -> None:
    # ToolNode reads its runtime from the graph config, so run it inside one.
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([create_load_skill_tool(skill_set)]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    call = {"name": "load_skill", "args": {"name": "finance/nope"}, "id": "call-1"}
    request = AIMessage(content="", tool_calls=[call])
    result = graph.compile().invoke({"messages": [request]})
    [_, message] = result["messages"]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert message.tool_call_id == "call-1"
    assert "finance/nope" in str(message.content)


def test_tool_unknown_path_lists_files(skill_set: SkillSet) -> None:
    out = _call(
        create_load_skill_tool(skill_set), name="finance/invoice-triage", path="nope.md"
    )
    assert out == (
        "Skill 'finance/invoice-triage' has no file 'nope.md'. "
        "Files: SKILL.md, reference/vendors.md"
    )


def test_tool_cleaned_up_set_reports_unavailable(skill_set: SkillSet) -> None:
    tool = create_load_skill_tool(skill_set)
    skill_set.root = None
    out = _call(tool, name="finance/invoice-triage")
    assert out.startswith("Skill 'finance/invoice-triage' is not available any more: ")


def test_tool_unreadable_file_reports_without_the_path(skill_set: SkillSet) -> None:
    tool = create_load_skill_tool(skill_set)
    _file(skill_set, "finance/invoice-triage").unlink()
    out = _call(tool, name="finance/invoice-triage")
    assert out == (
        "Skill 'finance/invoice-triage' could not be read: No such file or directory"
    )
    assert str(skill_set.root) not in out


def test_tool_truncates_long_text(
    skill_set: SkillSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "MAX_OUTPUT_CHARS", 10)
    vendors = _file(skill_set, "finance/invoice-triage", "reference/vendors.md")
    vendors.write_text("0123456789abcdef", encoding="utf-8")
    out = create_load_skill_tool(skill_set).invoke(
        {"name": "finance/invoice-triage", "path": "reference/vendors.md"}
    )
    assert out.endswith("\n\n0123456789\n\n[truncated: showing 10 of 16 characters]")


def test_tool_refuses_a_binary_file(skill_set: SkillSet) -> None:
    vendors = _file(skill_set, "finance/invoice-triage", "reference/vendors.md")
    vendors.write_bytes(b"PK\x03\x04\x00\x00binary")
    out = create_load_skill_tool(skill_set).invoke(
        {"name": "finance/invoice-triage", "path": "reference/vendors.md"}
    )
    assert out == (
        "Skill 'finance/invoice-triage' file 'reference/vendors.md' is not a text "
        "file and cannot be shown."
    )


def test_tool_name_and_metadata(skill_set: SkillSet) -> None:
    tool = create_load_skill_tool(skill_set)
    assert tool.name == "load_skill"
    assert tool.metadata == {"tool_type": "internal", "display_name": "load_skill"}
    renamed = create_load_skill_tool(skill_set, tool_name="read_skill")
    assert renamed.name == "read_skill"
    assert renamed.metadata == {"tool_type": "internal", "display_name": "read_skill"}
    assert "name" in _schema(renamed)["properties"]


@pytest.mark.parametrize("bad", ["", "load skill", "a/b", "x" * 65, "ok\n"])
def test_tool_rejects_an_invalid_tool_name(skill_set: SkillSet, bad: str) -> None:
    with pytest.raises(ValueError, match="tool_name"):
        create_load_skill_tool(skill_set, tool_name=bad)


def test_tool_with_empty_set_accepts_any_name_and_reports_none(
    empty_skill_set: SkillSet,
) -> None:
    tool = create_load_skill_tool(empty_skill_set)
    name = _schema(tool)["properties"]["name"]
    assert "enum" not in name
    assert name["type"] == "string"
    assert _call(tool, name="x") == "Unknown skill 'x'. Available: none"


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


async def test_tool_async_records_one_skill_load_span(
    skill_set: SkillSet, exporter: InMemorySpanExporter
) -> None:
    tool = create_load_skill_tool(skill_set)
    exporter.clear()
    await tool.ainvoke({"name": "finance/invoice-triage"})
    names = [span.name for span in exporter.get_finished_spans()]
    assert names == ["Skill load - finance/invoice-triage"]
