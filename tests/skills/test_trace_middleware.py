"""Tests for the span emitted when the model reads a SKILL.md."""

import logging
from typing import Any

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from uipath.platform.skills import SkillSet

from uipath_langchain.skills.middleware import SkillLoadTraceMiddleware

SKILL_MD = "/skills/finance/vendor-lookup/SKILL.md"


def _request(name: str, args: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "c1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )


@pytest.fixture
def mw(skill_set: SkillSet) -> SkillLoadTraceMiddleware:
    """finance v3 and ops v1 both have ``vendor-lookup``."""
    return SkillLoadTraceMiddleware(skill_set)


async def _ok(_: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id="c1")


async def test_read_of_a_skill_md_emits_a_span(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    async def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="x" * 42, tool_call_id="c1")

    result = await mw.awrap_tool_call(
        _request("read_file", {"file_path": SKILL_MD}), handler
    )

    assert isinstance(result, ToolMessage)
    (span,) = exporter.get_finished_spans()
    assert span.name == "Skill load - finance/vendor-lookup"
    assert span.attributes is not None
    assert span.attributes["skill.bundle.name"] == "finance"
    assert span.attributes["skill.name"] == "vendor-lookup"
    assert span.attributes["skill.version"] == "3"
    assert span.attributes["skill.invoked_by"] == "model"
    assert span.attributes["skill.bytes"] == 42
    assert span.attributes["uipath.custom_instrumentation"] is True


async def test_same_skill_name_in_two_bundles_is_attributed_to_the_right_bundle(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:

    await mw.awrap_tool_call(
        _request("read_file", {"file_path": "/skills/ops/vendor-lookup/SKILL.md"}),
        _ok,
    )
    await mw.awrap_tool_call(
        _request("read_file", {"file_path": "skills/finance/vendor-lookup/SKILL.md"}),
        _ok,
    )

    names = [
        (s.attributes or {})["skill.bundle.name"] for s in exporter.get_finished_spans()
    ]
    assert names == ["ops", "finance"]


async def test_other_reads_and_tools_emit_nothing(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:

    await mw.awrap_tool_call(_request("read_file", {"file_path": "/notes.md"}), _ok)
    await mw.awrap_tool_call(
        _request(
            "read_file",
            {"file_path": "/skills/finance/vendor-lookup/reference/vendors.md"},
        ),
        _ok,
    )
    await mw.awrap_tool_call(
        _request("read_file", {"file_path": "/skills/finance/unknown/SKILL.md"}), _ok
    )
    await mw.awrap_tool_call(
        _request("write_file", {"file_path": SKILL_MD, "content": ""}), _ok
    )
    await mw.awrap_tool_call(_request("read_file", {}), _ok)

    assert exporter.get_finished_spans() == ()


async def test_failed_read_emits_nothing(
    mw: SkillLoadTraceMiddleware, exporter: InMemorySpanExporter
) -> None:
    async def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="File not found", tool_call_id="c1", status="error")

    await mw.awrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), handler)

    assert exporter.get_finished_spans() == ()


def test_sync_path_emits_a_span(
    mw: SkillLoadTraceMiddleware, exporter: InMemorySpanExporter
) -> None:
    def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), handler)

    assert len(exporter.get_finished_spans()) == 1


async def test_command_results_are_passed_through_untouched(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    command: Command[Any] = Command(update={"messages": []})

    async def handler(_: ToolCallRequest) -> Command[Any]:
        return command

    result = await mw.awrap_tool_call(
        _request("read_file", {"file_path": SKILL_MD}), handler
    )

    assert result is command
    assert exporter.get_finished_spans() == ()


async def test_list_content_is_measured_by_its_text(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    async def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=[{"type": "text", "text": "abc"}, {"type": "text", "text": "de"}],
            tool_call_id="c1",
        )

    result = await mw.awrap_tool_call(
        _request("read_file", {"file_path": SKILL_MD}), handler
    )

    assert isinstance(result, ToolMessage)
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["skill.bytes"] == 5


async def test_bytes_count_utf8_bytes_not_characters(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    async def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="caf\u00e9", tool_call_id="c1")

    await mw.awrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), handler)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["skill.bytes"] == 5


def _break_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("telemetry down")

    monkeypatch.setattr("uipath_langchain.skills.middleware.record_skill_loaded", boom)


@pytest.mark.usefixtures("exporter")
async def test_telemetry_failure_never_breaks_the_tool_call(
    mw: SkillLoadTraceMiddleware,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _break_telemetry(monkeypatch)

    result = await mw.awrap_tool_call(
        _request("read_file", {"file_path": SKILL_MD}), _ok
    )

    assert isinstance(result, ToolMessage)
    assert result.content == "ok"


@pytest.mark.usefixtures("exporter")
async def test_telemetry_failure_warns_once(
    mw: SkillLoadTraceMiddleware,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _break_telemetry(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="uipath_langchain.skills.middleware"):
        for _ in range(2):
            await mw.awrap_tool_call(
                _request("read_file", {"file_path": SKILL_MD}), _ok
            )

    failures = [r for r in caplog.records if r.message == "Skill load tracing failed"]
    assert [r.levelno for r in failures] == [logging.WARNING, logging.DEBUG]


async def test_only_the_first_page_of_a_skill_md_emits_a_span(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:

    await mw.awrap_tool_call(
        _request("read_file", {"file_path": SKILL_MD, "offset": 100}), _ok
    )
    assert exporter.get_finished_spans() == ()

    await mw.awrap_tool_call(
        _request("read_file", {"file_path": SKILL_MD, "offset": 0}), _ok
    )
    await mw.awrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), _ok)
    assert len(exporter.get_finished_spans()) == 2


async def test_backslash_paths_are_normalized_like_the_tool(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    path = chr(92).join(["", "skills", "finance", "vendor-lookup", "SKILL.md"])

    await mw.awrap_tool_call(_request("read_file", {"file_path": path}), _ok)

    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["skill.bundle.name"] == "finance"


async def test_tool_exceptions_propagate_without_a_span(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    async def handler(_: ToolCallRequest) -> ToolMessage:
        raise OSError("disk gone")

    with pytest.raises(OSError, match="disk gone"):
        await mw.awrap_tool_call(
            _request("read_file", {"file_path": SKILL_MD}), handler
        )

    assert exporter.get_finished_spans() == ()
