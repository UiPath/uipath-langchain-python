"""Exporter wrapper that injects environment attributes into App Insights telemetry.

Adds AUTOMATION_SUITE_CLUSTER_ID and AUTOMATION_SUITE_CLUSTER_VERSION as
customDimensions on all App Insights telemetry types.
"""

import os
from collections.abc import Sequence

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from uipath_agents._observability.exporters.wrapped_span import WrappedSpan

_ENV_ATTRIBUTE_MAP: dict[str, str] = {
    "AUTOMATION_SUITE_CLUSTER_ID": "AutomationSuiteClusterId",
    "AUTOMATION_SUITE_CLUSTER_VERSION": "AutomationSuiteClusterVersion",
}


def get_env_attributes() -> dict[str, str]:
    attributes: dict[str, str] = {}
    for env_var, attr_key in _ENV_ATTRIBUTE_MAP.items():
        value = os.getenv(env_var)
        if value:
            attributes[attr_key] = value
    return attributes


class EnvironmentAttributesExporter(SpanExporter):
    """Wraps a SpanExporter to inject environment attributes into spans and events.

    Enriches both span-level attributes (for requests/dependencies) and
    event attributes (for the exceptions table) so the values appear as
    customDimensions across all App Insights telemetry types.
    """

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate
        self._attributes = get_env_attributes()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not self._attributes:
            return self._delegate.export(spans)

        enriched_spans = [self._enrich_span(span) for span in spans]
        return self._delegate.export(enriched_spans)

    def _enrich_span(self, span: ReadableSpan) -> ReadableSpan:
        enriched_attrs = dict(span.attributes) if span.attributes else {}
        enriched_attrs.update(self._attributes)

        enriched_events = [
            Event(
                name=event.name,
                attributes={**(event.attributes or {}), **self._attributes},
                timestamp=event.timestamp,
            )
            for event in span.events or []
        ]

        return WrappedSpan(span, enriched_attrs, enriched_events)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)
