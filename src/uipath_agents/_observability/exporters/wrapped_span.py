"""ReadableSpan wrapper that allows overriding attributes and events."""

from collections.abc import Sequence

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.util.types import Attributes


class WrappedSpan(ReadableSpan):
    """Wrapper around ReadableSpan with overridden attributes and events.

    Used by exporters that need to modify span data before export
    (e.g., PII redaction, attribute enrichment).
    """

    def __init__(
        self,
        span: ReadableSpan,
        attributes: Attributes,
        events: Sequence[Event],
    ) -> None:
        self._span = span
        self._attributes = attributes
        self._events = events

    @property
    def name(self) -> str:
        return self._span.name

    @property
    def context(self):
        return self._span.context

    @property
    def parent(self):
        return self._span.parent

    @property
    def start_time(self) -> int:
        return self._span.start_time  # type: ignore[return-value]

    @property
    def end_time(self) -> int:
        return self._span.end_time  # type: ignore[return-value]

    @property
    def status(self):
        return self._span.status

    @property
    def attributes(self) -> Attributes:
        return self._attributes

    @property
    def events(self) -> Sequence[Event]:
        return self._events

    @property
    def links(self):
        return self._span.links

    @property
    def kind(self):
        return self._span.kind

    @property
    def resource(self):
        return self._span.resource

    @property
    def instrumentation_scope(self):
        return self._span.instrumentation_scope

    @property
    def instrumentation_info(self):
        return self._span.instrumentation_info
