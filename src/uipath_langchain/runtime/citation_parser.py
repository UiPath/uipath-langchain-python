"""Stateless parser for <uip:cite .../> tags in LLM output."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Matches self-closing <uip:cite ... /> tags with key="value" attributes
_TAG_RE = re.compile(
    r'<uip:cite\s+((?:[a-z_]+="[^"]*"\s*)+)/\s*>'
)
_ATTR_RE = re.compile(r'([a-z_]+)="([^"]*)"')

# Prefix used for partial-tag detection
_TAG_PREFIX = "<uip:cite "


@dataclass(frozen=True)
class ParsedCitation:
    """A citation extracted from a <uip:cite .../> tag.

    Frozen for hashability so it can be used as a dict key for source dedup.
    """

    title: str
    url: str | None = None
    reference: str | None = None
    page_number: str | None = None


def parse_citations(text: str) -> list[tuple[str, ParsedCitation | None]]:
    """Parse text containing <uip:cite .../> tags into segments.

    Returns a list of (text_segment, citation_or_none) pairs.
    Text segments before/between/after tags are paired with None.
    A tag is paired with the text segment immediately preceding it.

    Invalid tags (neither url nor reference, or both) are stripped
    from text and logged as warnings.
    """
    result: list[tuple[str, ParsedCitation | None]] = []
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
            citation = ParsedCitation(
                title=title, url=url, page_number=page_number
            )
        elif has_reference and not has_url:
            citation = ParsedCitation(
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


def find_partial_tag_start(text: str) -> int:
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
