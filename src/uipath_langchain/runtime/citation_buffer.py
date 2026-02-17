"""Streaming buffer that parses <uip:cite .../> tags from LLM output chunks."""

from __future__ import annotations

from uuid import uuid4

from uipath.core.chat import (
    ParsedCitation,
    UiPathConversationCitationEndEvent,
    UiPathConversationCitationEvent,
    UiPathConversationCitationSourceMedia,
    UiPathConversationCitationSourceUrl,
    UiPathConversationCitationStartEvent,
    UiPathConversationContentPartChunkEvent,
    find_partial_tag_start,
    parse_citations,
)


class CitationStreamBuffer:
    """Buffer that accumulates streaming text, parses citation tags, and emits chunk events.

    Maintains state per message stream for source dedup and partial tag buffering.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._source_numbers: dict[ParsedCitation, int] = {}
        self._next_number: int = 1

    def _get_source_number(self, citation: ParsedCitation) -> int:
        """Get or assign a source number for dedup (matching C# record equality)."""
        if citation not in self._source_numbers:
            self._source_numbers[citation] = self._next_number
            self._next_number += 1
        return self._source_numbers[citation]

    def _build_source(
        self, citation: ParsedCitation, number: int
    ) -> UiPathConversationCitationSourceUrl | UiPathConversationCitationSourceMedia:
        """Map a ParsedCitation to the appropriate source model."""
        if citation.url is not None:
            return UiPathConversationCitationSourceUrl(
                title=citation.title,
                number=number,
                url=citation.url,
            )
        # reference-based (media) citation
        return UiPathConversationCitationSourceMedia(
            title=citation.title,
            number=number,
            mime_type=None,
            download_url=citation.reference,
            page_number=citation.page_number,
        )

    def _make_chunk(
        self,
        text: str,
        citation: ParsedCitation | None = None,
    ) -> UiPathConversationContentPartChunkEvent:
        """Create a content part chunk event, optionally with a citation."""
        if citation is None:
            return UiPathConversationContentPartChunkEvent(data=text)

        number = self._get_source_number(citation)
        source = self._build_source(citation, number)

        return UiPathConversationContentPartChunkEvent(
            data=text,
            citation=UiPathConversationCitationEvent(
                citation_id=str(uuid4()),
                start=UiPathConversationCitationStartEvent(),
                end=UiPathConversationCitationEndEvent(sources=[source]),
            ),
        )

    def add_chunk(self, text: str) -> list[UiPathConversationContentPartChunkEvent]:
        """Process a new text chunk from the LLM stream.

        Returns a list of chunk events to emit. Tags are parsed and stripped,
        partial tags at the end are held back until more data arrives.
        """
        self._buffer += text

        # Check for a partial tag at the end
        partial_idx = find_partial_tag_start(self._buffer)
        if partial_idx >= 0:
            processable = self._buffer[:partial_idx]
            self._buffer = self._buffer[partial_idx:]
        else:
            processable = self._buffer
            self._buffer = ""

        if not processable:
            return []

        segments = parse_citations(processable)
        if not segments:
            return []

        # Build events with text coalescing: merge adjacent plain-text chunks
        events: list[UiPathConversationContentPartChunkEvent] = []
        for segment_text, citation in segments:
            if citation is not None:
                events.append(self._make_chunk(segment_text, citation))
            elif segment_text:
                # Plain text — coalesce with previous plain-text event
                if events and events[-1].citation is None and events[-1].data is not None:
                    events[-1] = UiPathConversationContentPartChunkEvent(
                        data=events[-1].data + segment_text
                    )
                else:
                    events.append(self._make_chunk(segment_text))

        return events

    def finalize(self) -> list[UiPathConversationContentPartChunkEvent]:
        """Flush remaining buffer as plain text.

        Any partial tag at end of stream becomes literal text (graceful degradation).
        """
        if not self._buffer:
            return []

        remaining = self._buffer
        self._buffer = ""
        return [self._make_chunk(remaining)]
