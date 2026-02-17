"""Streaming buffer that parses <uip:cite .../> tags from LLM output chunks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from uuid import uuid4

from uipath.core.chat import (
    UiPathConversationCitationEndEvent,
    UiPathConversationCitationEvent,
    UiPathConversationCitationSourceMedia,
    UiPathConversationCitationSourceUrl,
    UiPathConversationCitationStartEvent,
    UiPathConversationContentPartChunkEvent,
)

logger = logging.getLogger(__name__)

# Matches self-closing <uip:cite ... /> tags with key="value" attributes
_TAG_RE = re.compile(r'<uip:cite\s+((?:[a-z_]+="[^"]*"\s*)+)/\s*>')
_ATTR_RE = re.compile(r'([a-z_]+)="([^"]*)"')

# Prefix used for partial-tag detection
_TAG_PREFIX = "<uip:cite "


@dataclass(frozen=True)
class _ParsedCitation:
    """A citation extracted from a <uip:cite .../> tag.

    Frozen for hashability so it can be used as a dict key for source dedup.
    """

    title: str
    url: str | None = None
    reference: str | None = None
    page_number: str | None = None


def _parse_citations(text: str) -> list[tuple[str, _ParsedCitation | None]]:
    """Parse text containing <uip:cite .../> tags into segments.

    Returns a list of (text_segment, citation_or_none) pairs.
    Text segments before/between/after tags are paired with None.
    A tag is paired with the text segment immediately preceding it.

    Invalid tags (neither url nor reference, or both) are stripped
    from text and logged as warnings.
    """
    result: list[tuple[str, _ParsedCitation | None]] = []
    last_end = 0

    for match in _TAG_RE.finditer(text):
        # Text before this tag
        preceding_text = text[last_end : match.start()]
        attrs_str = match.group(1)
        attrs = dict(_ATTR_RE.findall(attrs_str))

        title = attrs.get("title", "")
        url = attrs.get("url")
        reference = attrs.get("reference")
        page_number = attrs.get("page_number")

        has_url = url is not None
        has_reference = reference is not None

        if has_url and not has_reference:
            citation = _ParsedCitation(
                title=title, url=url, page_number=page_number
            )
        elif has_reference and not has_url:
            citation = _ParsedCitation(
                title=title, reference=reference, page_number=page_number
            )
        else:
            # Invalid: both or neither — strip tag, log warning
            logger.warning(
                "Invalid <uip:cite/> tag (needs exactly one of url or reference): %s",
                match.group(0),
            )
            # Emit preceding text as plain, skip the tag
            if preceding_text:
                result.append((preceding_text, None))
            last_end = match.end()
            continue

        # Emit text with its citation
        if preceding_text:
            result.append((preceding_text, citation))
        else:
            # Citation with no preceding text — emit empty string with citation
            result.append(("", citation))

        last_end = match.end()

    # Remaining text after last tag
    remaining = text[last_end:]
    if remaining:
        result.append((remaining, None))

    return result


def _find_partial_tag_start(text: str) -> int:
    """Find the index of a trailing '<' that could be the start of a <uip:cite tag.

    Returns the index if found, or -1 if no partial tag is detected.
    This handles cases like '<', '<u', '<uip:cite title="partial...' where
    the full self-closing tag hasn't been received yet.
    """
    # Search backwards for the last '<' that isn't part of a complete tag
    idx = text.rfind("<")
    if idx == -1:
        return -1

    suffix = text[idx:]

    # If there's a complete self-closing tag ending with /> at or after idx,
    # this '<' is consumed — no partial tag
    if "/>" in suffix:
        return -1

    # Short suffix: check if it's a prefix of "<uip:cite "
    if len(suffix) <= len(_TAG_PREFIX):
        if _TAG_PREFIX.startswith(suffix):
            return idx
        return -1

    # Longer suffix: starts with "<uip:cite " but no closing "/>"
    if suffix.startswith(_TAG_PREFIX):
        return idx

    return -1


class CitationStreamBuffer:
    """Buffer that accumulates streaming text, parses citation tags, and emits chunk events.

    Maintains state per message stream for source dedup and partial tag buffering.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._source_numbers: dict[_ParsedCitation, int] = {}
        self._next_number: int = 1

    def _get_source_number(self, citation: _ParsedCitation) -> int:
        """Get or assign a source number for dedup (matching C# record equality)."""
        if citation not in self._source_numbers:
            self._source_numbers[citation] = self._next_number
            self._next_number += 1
        return self._source_numbers[citation]

    def _build_source(
        self, citation: _ParsedCitation, number: int
    ) -> UiPathConversationCitationSourceUrl | UiPathConversationCitationSourceMedia:
        """Map a _ParsedCitation to the appropriate source model."""
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
        citation: _ParsedCitation | None = None,
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

        Returns chunk events to emit. Tags are parsed and stripped,
        partial tags at the end are held back until more data arrives.
        """
        self._buffer += text

        # Check for a partial tag at the end
        partial_idx = _find_partial_tag_start(self._buffer)
        if partial_idx >= 0:
            processable = self._buffer[:partial_idx]
            self._buffer = self._buffer[partial_idx:]
        else:
            processable = self._buffer
            self._buffer = ""

        if not processable:
            return []

        segments = _parse_citations(processable)
        if not segments:
            return []

        # Build events with text coalescing: merge adjacent plain-text chunks
        chunks: list[UiPathConversationContentPartChunkEvent] = []
        for segment_text, citation in segments:
            if citation is not None:
                chunks.append(self._make_chunk(segment_text, citation))
            elif segment_text:
                # Plain text — coalesce with previous plain-text chunk
                if chunks and chunks[-1].citation is None and chunks[-1].data is not None:
                    chunks[-1] = UiPathConversationContentPartChunkEvent(
                        data=chunks[-1].data + segment_text
                    )
                else:
                    chunks.append(self._make_chunk(segment_text))

        return chunks

    def finalize(self) -> list[UiPathConversationContentPartChunkEvent]:
        """Flush remaining buffer as plain text.

        Any partial tag at end of stream becomes literal text (graceful degradation).
        """
        if not self._buffer:
            return []

        remaining = self._buffer
        self._buffer = ""
        return [self._make_chunk(remaining)]
