"""Tests for OpenTelemetry utility helpers."""

import builtins

from uipath_langchain._utils._otel import (
    get_current_span_and_trace_ids,
    set_current_span_error,
    set_span_attribute,
)


class _SpanContext:
    is_valid = True
    span_id = 0x123
    trace_id = 0x456


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.exceptions: list[BaseException] = []
        self.status: tuple[object, str] | None = None

    def get_span_context(self) -> _SpanContext:
        return _SpanContext()

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, name: str, value: object) -> None:
        self.attributes[name] = value

    def record_exception(self, error: BaseException) -> None:
        self.exceptions.append(error)

    def set_status(self, code: object, description: str) -> None:
        self.status = (code, description)


class _InvalidSpan(_RecordingSpan):
    def get_span_context(self):
        class InvalidContext:
            is_valid = False

        return InvalidContext()


def test_get_current_span_and_trace_ids(monkeypatch) -> None:
    from opentelemetry import trace

    monkeypatch.setattr(trace, "get_current_span", lambda: _RecordingSpan())

    assert get_current_span_and_trace_ids() == (
        "0000000000000123",
        "00000000000000000000000000000456",
    )


def test_get_current_span_and_trace_ids_returns_empty_for_invalid_context(
    monkeypatch,
) -> None:
    from opentelemetry import trace

    monkeypatch.setattr(trace, "get_current_span", lambda: _InvalidSpan())

    assert get_current_span_and_trace_ids() == ("", "")


def test_set_span_attribute(monkeypatch) -> None:
    from opentelemetry import trace

    span = _RecordingSpan()
    monkeypatch.setattr(trace, "get_current_span", lambda: span)

    set_span_attribute("savedToMemory", True)

    assert span.attributes == {"savedToMemory": True}


def test_set_current_span_error(monkeypatch) -> None:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    span = _RecordingSpan()
    error = RuntimeError("memory failed")
    monkeypatch.setattr(trace, "get_current_span", lambda: span)

    set_current_span_error(error)

    assert span.exceptions == [error]
    assert span.status == (StatusCode.ERROR, "memory failed")


def test_otel_helpers_are_noops_without_opentelemetry(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert get_current_span_and_trace_ids() == ("", "")
    set_span_attribute("fromMemory", False)
    set_current_span_error(RuntimeError("memory failed"))
