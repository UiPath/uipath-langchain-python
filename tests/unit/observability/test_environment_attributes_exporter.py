"""Tests for environment attributes exporter."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from uipath_agents._observability.exporters.environment_attributes_exporter import (
    EnvironmentAttributesExporter,
    get_env_attributes,
)
from uipath_agents._observability.exporters.wrapped_span import WrappedSpan


class TestGetEnvAttributes:
    def test_returns_empty_when_env_vars_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_env_attributes() == {}

    def test_returns_cluster_id_when_set(self) -> None:
        with patch.dict(
            os.environ,
            {"AUTOMATION_SUITE_CLUSTER_ID": "cluster-1"},
            clear=True,
        ):
            result = get_env_attributes()
            assert result == {"AutomationSuiteClusterId": "cluster-1"}

    def test_returns_cluster_version_when_set(self) -> None:
        with patch.dict(
            os.environ,
            {"AUTOMATION_SUITE_CLUSTER_VERSION": "24.10"},
            clear=True,
        ):
            result = get_env_attributes()
            assert result == {"AutomationSuiteClusterVersion": "24.10"}

    def test_returns_both_when_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AUTOMATION_SUITE_CLUSTER_ID": "cluster-1",
                "AUTOMATION_SUITE_CLUSTER_VERSION": "24.10",
            },
            clear=True,
        ):
            result = get_env_attributes()
            assert result == {
                "AutomationSuiteClusterId": "cluster-1",
                "AutomationSuiteClusterVersion": "24.10",
            }

    def test_skips_empty_string_values(self) -> None:
        with patch.dict(
            os.environ,
            {"AUTOMATION_SUITE_CLUSTER_ID": "", "AUTOMATION_SUITE_CLUSTER_VERSION": ""},
            clear=True,
        ):
            assert get_env_attributes() == {}


def _make_span(
    attributes: dict[str, str] | None = None,
    events: list[Event] | None = None,
) -> Mock:
    span = Mock(spec=ReadableSpan)
    span.attributes = attributes
    span.events = events or []
    span.name = "test-span"
    span.start_time = 100
    span.end_time = 200
    return span


@pytest.fixture
def env_vars() -> dict[str, str]:
    return {
        "AUTOMATION_SUITE_CLUSTER_ID": "cluster-1",
        "AUTOMATION_SUITE_CLUSTER_VERSION": "24.10",
    }


class TestEnvironmentAttributesExporter:
    def test_passthrough_when_no_env_vars_set(self) -> None:
        delegate = MagicMock()
        delegate.export.return_value = SpanExportResult.SUCCESS
        span = _make_span(attributes={"key": "value"})

        with patch.dict(os.environ, {}, clear=True):
            exporter = EnvironmentAttributesExporter(delegate)
            result = exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        delegate.export.assert_called_once_with([span])

    def test_enriches_span_attributes(self, env_vars: dict[str, str]) -> None:
        delegate = MagicMock()
        delegate.export.return_value = SpanExportResult.SUCCESS
        span = _make_span(attributes={"existing": "attr"})

        with patch.dict(os.environ, env_vars, clear=True):
            exporter = EnvironmentAttributesExporter(delegate)
            exporter.export([span])

        exported = delegate.export.call_args[0][0]
        attrs = exported[0].attributes
        assert attrs["existing"] == "attr"
        assert attrs["AutomationSuiteClusterId"] == "cluster-1"
        assert attrs["AutomationSuiteClusterVersion"] == "24.10"

    def test_enriches_exception_event_attributes(
        self, env_vars: dict[str, str]
    ) -> None:
        delegate = MagicMock()
        delegate.export.return_value = SpanExportResult.SUCCESS
        exception_event = Event(
            name="exception",
            attributes={"exception.type": "ValueError", "exception.message": "bad"},
            timestamp=123,
        )
        span = _make_span(events=[exception_event])

        with patch.dict(os.environ, env_vars, clear=True):
            exporter = EnvironmentAttributesExporter(delegate)
            exporter.export([span])

        exported = delegate.export.call_args[0][0]
        event_attrs = exported[0].events[0].attributes
        assert event_attrs["exception.type"] == "ValueError"
        assert event_attrs["exception.message"] == "bad"
        assert event_attrs["AutomationSuiteClusterId"] == "cluster-1"
        assert event_attrs["AutomationSuiteClusterVersion"] == "24.10"

    def test_handles_span_with_no_attributes(self, env_vars: dict[str, str]) -> None:
        delegate = MagicMock()
        delegate.export.return_value = SpanExportResult.SUCCESS
        span = _make_span(attributes=None)

        with patch.dict(os.environ, env_vars, clear=True):
            exporter = EnvironmentAttributesExporter(delegate)
            exporter.export([span])

        exported = delegate.export.call_args[0][0]
        attrs = exported[0].attributes
        assert attrs["AutomationSuiteClusterId"] == "cluster-1"

    def test_handles_span_with_no_events(self, env_vars: dict[str, str]) -> None:
        delegate = MagicMock()
        delegate.export.return_value = SpanExportResult.SUCCESS
        span = _make_span(attributes={"key": "val"}, events=None)
        span.events = None

        with patch.dict(os.environ, env_vars, clear=True):
            exporter = EnvironmentAttributesExporter(delegate)
            exporter.export([span])

        exported = delegate.export.call_args[0][0]
        assert exported[0].events == []

    def test_returns_wrapped_span(self, env_vars: dict[str, str]) -> None:
        delegate = MagicMock()
        delegate.export.return_value = SpanExportResult.SUCCESS
        span = _make_span(attributes={"key": "val"})

        with patch.dict(os.environ, env_vars, clear=True):
            exporter = EnvironmentAttributesExporter(delegate)
            exporter.export([span])

        exported = delegate.export.call_args[0][0]
        assert isinstance(exported[0], WrappedSpan)

    def test_empty_span_list(self, env_vars: dict[str, str]) -> None:
        delegate = MagicMock()
        delegate.export.return_value = SpanExportResult.SUCCESS

        with patch.dict(os.environ, env_vars, clear=True):
            exporter = EnvironmentAttributesExporter(delegate)
            result = exporter.export([])

        assert result == SpanExportResult.SUCCESS
        delegate.export.assert_called_once_with([])

    def test_shutdown_delegates(self) -> None:
        delegate = MagicMock()
        exporter = EnvironmentAttributesExporter(delegate)

        exporter.shutdown()

        delegate.shutdown.assert_called_once()

    def test_force_flush_delegates(self) -> None:
        delegate = MagicMock()
        delegate.force_flush.return_value = True
        exporter = EnvironmentAttributesExporter(delegate)

        result = exporter.force_flush(5000)

        assert result is True
        delegate.force_flush.assert_called_once_with(5000)
