from __future__ import annotations

import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from uipath.core.chat import (
    UiPathConversationCitationData,
    UiPathConversationCitationEndEvent,
    UiPathConversationCitationEvent,
    UiPathConversationCitationSourceMedia,
    UiPathConversationCitationSourceUrl,
    UiPathConversationCitationStartEvent,
    UiPathConversationContentPartChunkEvent,
)

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r'<uip:cite\s+((?:[a-z_]+="(?:[^"\\]|\\.)*"\s*)+)/\s*>')
_ATTR_RE = re.compile(r'([a-z_]+)="((?:[^"\\]|\\.)*)"')


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
        attributes = {
            k: v.replace('\\"', '"') for k, v in _ATTR_RE.findall(raw_attributes)
        }

        title = attributes.get("title", "")
        url = attributes.get("url")
        reference = attributes.get("reference")
        page_number = attributes.get("page_number")

        has_url = url is not None
        has_reference = reference is not None
        has_page_number = page_number is not None

        if has_url and not has_reference:
            # web citation
            citation = _ParsedCitation(title=title, url=url, page_number=page_number)
        elif has_reference and has_page_number and not has_url:
            # context grounding citation
            citation = _ParsedCitation(
                title=title, reference=reference, page_number=page_number
            )
        else:
            # skip; doesn't match a valid source type
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


def _make_source(
    citation: _ParsedCitation,
    source_numbers: dict[_ParsedCitation, int],
    next_number: int,
) -> tuple[
    UiPathConversationCitationSourceUrl | UiPathConversationCitationSourceMedia | None,
    int,
]:
    """Build a citation source, deduplicating by assigning numbers."""
    if citation.url is not None:
        if citation not in source_numbers:
            source_numbers[citation] = next_number
            next_number += 1
        return UiPathConversationCitationSourceUrl(
            title=citation.title,
            number=source_numbers[citation],
            url=citation.url,
        ), next_number
    elif citation.reference is not None and citation.page_number is not None:
        if citation not in source_numbers:
            source_numbers[citation] = next_number
            next_number += 1
        mime_type, _ = mimetypes.guess_type(citation.title)
        return UiPathConversationCitationSourceMedia(
            title=citation.title,
            number=source_numbers[citation],
            mime_type=mime_type,
            download_url=citation.reference,
            page_number=citation.page_number,
        ), next_number
    else:
        return None, next_number


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

    def _process_segments(
        self, text: str
    ) -> list[UiPathConversationContentPartChunkEvent]:
        segments = _parse_citations(text)
        if not segments:
            return []

        # Emit each segment as up to two chunk events: a plain text chunk for the
        # preceding text (if any), then a citation-only chunk with no data. CAS's
        # content-part-event-helper records offset on start and length on end; with
        # no data between them, length resolves to 0 (a point citation).
        content_parts: list[UiPathConversationContentPartChunkEvent] = []
        for segment_text, citation in segments:
            if segment_text:
                content_parts.append(
                    UiPathConversationContentPartChunkEvent(data=segment_text)
                )
            if citation is None:
                continue
            source, self._next_number = _make_source(
                citation, self._source_numbers, self._next_number
            )
            if source is None:
                continue
            content_parts.append(
                UiPathConversationContentPartChunkEvent(
                    citation=UiPathConversationCitationEvent(
                        citation_id=str(uuid4()),
                        start=UiPathConversationCitationStartEvent(),
                        end=UiPathConversationCitationEndEvent(sources=[source]),
                    ),
                )
            )

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


def extract_citations_from_text(
    text: str,
) -> tuple[str, list[UiPathConversationCitationData]]:
    """Parse inline <uip:cite .../> tags from text and return cleaned text with structured point citations.

    Each tag becomes a length=0 citation anchored at the offset where the tag appeared
    (i.e., immediately after its preceding text). This mirrors what the streaming emitter
    produces on the wire, where start/end fire around an empty data slice.
    """
    segments = _parse_citations(text)
    if not segments:
        return (text, [])

    source_numbers: dict[_ParsedCitation, int] = {}
    next_number = 1
    cleaned_parts: list[str] = []
    citations: list[UiPathConversationCitationData] = []
    offset = 0

    for segment_text, citation in segments:
        cleaned_parts.append(segment_text)
        offset += len(segment_text)

        if citation is None:
            continue
        source, next_number = _make_source(citation, source_numbers, next_number)
        if source is None:
            continue
        citations.append(
            UiPathConversationCitationData(
                offset=offset,
                length=0,
                sources=[source],
            )
        )

    return ("".join(cleaned_parts), citations)


def reconstruct_text_with_citations(
    cleaned_text: str,
    citations: Sequence[UiPathConversationCitationData] | None,
) -> str:
    """Inverse of extract_citations_from_text: reinsert <uip:cite/> tags into cleaned text.

    Used when replaying assistant messages so the LLM sees its own prior citation markup
    and doesn't over- or under-cite on the next turn. Citations are point anchors
    (length=0), so tags are inserted at the offset directly.
    """
    if not citations:
        return cleaned_text

    # Sort by offset so we never walk the cursor backwards — out-of-order
    # citations would otherwise re-emit text already added to `parts`.
    ordered = sorted(citations, key=lambda c: c.offset)

    parts: list[str] = []
    cursor = 0
    for citation in ordered:
        parts.append(cleaned_text[cursor : citation.offset])
        for source in citation.sources:
            parts.append(_source_to_cite_tag(source))
        cursor = citation.offset

    parts.append(cleaned_text[cursor:])
    return "".join(parts)


def _source_to_cite_tag(
    source: UiPathConversationCitationSourceUrl | UiPathConversationCitationSourceMedia,
) -> str:
    if isinstance(source, UiPathConversationCitationSourceUrl):
        return (
            f'<uip:cite title="{_escape_attr(source.title)}" '
            f'url="{_escape_attr(source.url)}" />'
        )
    if isinstance(source, UiPathConversationCitationSourceMedia):
        return (
            f'<uip:cite title="{_escape_attr(source.title)}" '
            f'reference="{_escape_attr(source.download_url or "")}" '
            f'page_number="{_escape_attr(source.page_number or "")}" />'
        )
    return ""


def _escape_attr(value: str) -> str:
    """Escape only characters that would break XML attribute parsing."""
    return value.replace('"', "&quot;")


def convert_citations_to_inline_tags(content: dict[str, Any]) -> str:
    """Replace [ordinal] references in DeepRag text with <uip:cite/> tags."""
    text = content.get("text", "")
    citations = content.get("citations", [])

    citation_map: dict[int, dict[str, Any]] = {}
    for c in citations:
        ordinal = c.get("ordinal")
        if ordinal is not None:
            citation_map[ordinal] = c

    for ordinal, c in citation_map.items():
        title = _escape_attr(str(c.get("source", "")))
        reference = _escape_attr(str(c.get("reference", "")))
        page_number = _escape_attr(str(c.get("pageNumber", c.get("page_number", ""))))
        tag = (
            f'<uip:cite title="{title}" '
            f'reference="{reference}" '
            f'page_number="{page_number}" />'
        )
        text = text.replace(f"[{ordinal}]", tag)

    return text


async def cas_deep_rag_citation_wrapper(tool: BaseTool, call: ToolCall):
    """Transform DeepRag results into CAS's inline <uip:cite/> tags."""
    result = await tool.ainvoke(call)
    try:
        data = json.loads(result.content)
        result.content = json.dumps({"text": convert_citations_to_inline_tags(data)})
    except Exception:
        logger.warning(
            "Failed to transform DeepRag citations, returning raw result", exc_info=True
        )
    return result
