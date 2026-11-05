"""Tests for the span emitted when the model reads a SKILL.md."""

import logging
from typing import Any

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from uipath.platform.skills import SkillSet, _telemetry

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


async def test_the_tool_runs_inside_the_skill_load_span(
    mw: SkillLoadTraceMiddleware, exporter: InMemorySpanExporter
) -> None:
    seen: list[trace.Span] = []

    async def handler(_: ToolCallRequest) -> ToolMessage:
        seen.append(trace.get_current_span())
        with _telemetry._tracer.start_as_current_span("Tool call - read_file"):
            pass
        return ToolMessage(content="ok", tool_call_id="c1")

    await mw.awrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), handler)

    child, parent = exporter.get_finished_spans()
    assert parent.name == "Skill load - finance/vendor-lookup"
    assert seen[0].get_span_context().span_id == parent.context.span_id
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id


async def test_other_tools_run_outside_any_skill_load_span(
    mw: SkillLoadTraceMiddleware, exporter: InMemorySpanExporter
) -> None:
    seen: list[trace.Span] = []

    async def handler(_: ToolCallRequest) -> ToolMessage:
        seen.append(trace.get_current_span())
        return ToolMessage(content="ok", tool_call_id="c1")

    await mw.awrap_tool_call(_request("read_file", {"file_path": "/notes.md"}), handler)

    assert seen == [trace.INVALID_SPAN]
    assert exporter.get_finished_spans() == ()


async def test_failed_read_leaves_an_error_span(
    mw: SkillLoadTraceMiddleware, exporter: InMemorySpanExporter
) -> None:
    async def handler(_: ToolCallRequest) -> ToolMessage:
        return ToolMessage(content="File not found", tool_call_id="c1", status="error")

    await mw.awrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), handler)

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "read_file returned an error"
    assert "skill.bytes" not in (span.attributes or {})


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
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "read_file returned no tool message"


async def test_a_command_carrying_the_tool_message_counts_as_a_load(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    async def handler(_: ToolCallRequest) -> Command[Any]:
        return Command(
            update={"messages": [ToolMessage(content="x" * 9, tool_call_id="c1")]}
        )

    await mw.awrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), handler)

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.UNSET
    assert (span.attributes or {})["skill.bytes"] == 9


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

    monkeypatch.setattr("uipath_langchain.skills.middleware.skill_load_span", boom)


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


async def test_tool_exceptions_propagate_and_are_recorded_on_the_span(
    mw: SkillLoadTraceMiddleware,
    exporter: InMemorySpanExporter,
) -> None:
    async def handler(_: ToolCallRequest) -> ToolMessage:
        raise OSError("disk gone")

    with pytest.raises(OSError, match="disk gone"):
        await mw.awrap_tool_call(
            _request("read_file", {"file_path": SKILL_MD}), handler
        )

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]


@pytest.mark.usefixtures("exporter")
async def test_telemetry_failure_still_runs_the_tool_once(
    mw: SkillLoadTraceMiddleware,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _break_telemetry(monkeypatch)
    calls: list[ToolCallRequest] = []

    async def handler(request: ToolCallRequest) -> ToolMessage:
        calls.append(request)
        return ToolMessage(content="ok", tool_call_id="c1")

    await mw.awrap_tool_call(_request("read_file", {"file_path": SKILL_MD}), handler)

    assert len(calls) == 1
