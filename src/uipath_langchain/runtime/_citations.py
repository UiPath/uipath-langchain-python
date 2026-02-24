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

_TAG_RE = re.compile(r'<uip:cite\s+((?:[a-z_]+="[^"]*"\s*)+)/\s*>')
_ATTR_RE = re.compile(r'([a-z_]+)="([^"]*)"')


@dataclass(frozen=True)  # frozen to make hashable / de-dupe sources
class _ParsedCitation:
    title: str
    url: str | None = None
    reference: str | None = None
    page_number: str | None = None


# <uip:cite .../> tags -> [(text_segment, citation_or_none)]
def _parse_citations(text: str) -> list[tuple[str, _ParsedCitation | None]]:
    segments: list[tuple[str, _ParsedCitation | None]] = []
    cursor = 0

    # find <uip:cite title="foo" url="https://..." />
    for match in _TAG_RE.finditer(text):
        preceding_text = text[cursor : match.start()]
        raw_attributes = match.group(1)

        # title="foo" url="https://..." -> [("title","foo"), ("url","https://...")]
        attributes = dict(_ATTR_RE.findall(raw_attributes))

        title = attributes.get("title", "")
        url = attributes.get("url")
        reference = attributes.get("reference")
        page_number = attributes.get("page_number")

        has_url = url is not None
        has_reference = reference is not None

        if has_url and not has_reference:
            # web citation
            citation = _ParsedCitation(title=title, url=url, page_number=page_number)
        elif has_reference and not has_url:
            # context grounding citation
            citation = _ParsedCitation(
                title=title, reference=reference, page_number=page_number
            )
        else:
            # skip; citation has no url= or reference=
            if preceding_text:
                segments.append((preceding_text, None))
            cursor = match.end()
            continue

        # Citation applies to the preceding text segment (e.g some text [citation])
        if preceding_text:
            segments.append((preceding_text, citation))
        else:
            # No preceding text (e.g. back-to-back citations)
            segments.append(("", citation))

        cursor = match.end()

    trailing_text = text[cursor:]
    if trailing_text:
        segments.append((trailing_text, None))

    return segments


def _find_partial_tag_start(text: str) -> int:
    _TAG_PREFIX = "<uip:cite "

    bracket_pos = text.rfind("<")
    if bracket_pos == -1:
        return -1

    suffix = text[bracket_pos:]

    # "<uip:cite title="some partial" />"
    if "/>" in suffix:
        return -1

    # "<", "<u", "<uip:", "<uip:cite"
    if len(suffix) <= len(_TAG_PREFIX):
        if _TAG_PREFIX.startswith(suffix):
            return bracket_pos
        return -1

    # "<uip:cite title="some partial"
    if suffix.startswith(_TAG_PREFIX):
        return bracket_pos

    return -1


class CitationStreamProcessor:
    def __init__(self) -> None:
        self._buffer: str = ""
        self._source_numbers: dict[_ParsedCitation, int] = {}
        self._next_number: int = 1

    def _build_content_part_citation(
        self,
        text: str,
        citation: _ParsedCitation | None = None,
    ) -> UiPathConversationContentPartChunkEvent:
        if citation is None:
            return UiPathConversationContentPartChunkEvent(data=text)

        if citation not in self._source_numbers:
            self._source_numbers[citation] = self._next_number
            self._next_number += 1
        number = self._source_numbers[citation]

        source: (
            UiPathConversationCitationSourceUrl | UiPathConversationCitationSourceMedia
        )
        if citation.url is not None:
            source = UiPathConversationCitationSourceUrl(
                title=citation.title,
                number=number,
                url=citation.url,
            )
        else:
            source = UiPathConversationCitationSourceMedia(
                title=citation.title,
                number=number,
                mime_type=None,
                download_url=citation.reference,
                page_number=citation.page_number,
            )

        return UiPathConversationContentPartChunkEvent(
            data=text,
            citation=UiPathConversationCitationEvent(
                citation_id=str(uuid4()),
                start=UiPathConversationCitationStartEvent(),
                end=UiPathConversationCitationEndEvent(sources=[source]),
            ),
        )

    def _process_segments(
        self, text: str
    ) -> list[UiPathConversationContentPartChunkEvent]:
        segments = _parse_citations(text)
        if not segments:
            return []

        content_parts: list[UiPathConversationContentPartChunkEvent] = []
        for segment_text, citation in segments:
            if citation is not None:
                content_part_with_citation = self._build_content_part_citation(
                    segment_text, citation
                )
                content_parts.append(content_part_with_citation)
            elif segment_text:
                content_part_plain = self._build_content_part_citation(segment_text)
                content_parts.append(content_part_plain)

        return content_parts

    def add_chunk(self, text: str) -> list[UiPathConversationContentPartChunkEvent]:
        self._buffer += text

        partial_tag_start = _find_partial_tag_start(self._buffer)
        if partial_tag_start >= 0:
            completed_text = self._buffer[:partial_tag_start]
            self._buffer = self._buffer[partial_tag_start:]
        else:
            completed_text = self._buffer
            self._buffer = ""

        if not completed_text:
            return []

        return self._process_segments(completed_text)

    # Flush remaining content parts / citations
    def finalize(self) -> list[UiPathConversationContentPartChunkEvent]:
        if not self._buffer:
            return []

        remaining = self._buffer
        self._buffer = ""

        return self._process_segments(remaining)
