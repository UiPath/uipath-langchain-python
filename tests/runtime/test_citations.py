"""Tests for the CitationStreamProcessor and citation parsing utilities."""
# mypy: disable-error-code="union-attr,operator"

import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages.tool import ToolCall, ToolMessage
from uipath.core.chat import (
    UiPathConversationCitationSourceMedia,
    UiPathConversationCitationSourceUrl,
)

from uipath_langchain.runtime._citations import (
    CitationStreamProcessor,
    _find_partial_tag_start,
    cas_deep_rag_citation_wrapper,
    convert_citations_to_inline_tags,
    extract_citations_from_text,
)


class TestFindPartialTagStart:
    def test_partial_tag_with_attributes_buffers(self):
        assert _find_partial_tag_start('text <uip:cite title="partial') == 5

    def test_non_matching_html_tag_does_not_buffer(self):
        assert _find_partial_tag_start("text <div>") == -1

    def test_no_angle_bracket(self):
        assert _find_partial_tag_start("plain text") == -1

    def test_empty_string(self):
        assert _find_partial_tag_start("") == -1

    def test_just_open_bracket(self):
        """Single '<' is a prefix of '<uip:cite ', so it should buffer."""
        assert _find_partial_tag_start("text <") == 5

    def test_partial_uip_prefix(self):
        """'<uip:cit' is a prefix of '<uip:cite ', should buffer."""
        assert _find_partial_tag_start("text <uip:cit") == 5

    def test_partial_uip_colon(self):
        """'<uip:' is a prefix of '<uip:cite ', should buffer."""
        assert _find_partial_tag_start("text <uip:") == 5

    def test_non_matching_uip_prefix(self):
        """'<uip:something' does NOT start with '<uip:cite ', should not buffer."""
        assert _find_partial_tag_start("text <uip:something") == -1

    def test_completed_tag_does_not_buffer(self):
        """A fully closed tag (with />) should not buffer."""
        assert (
            _find_partial_tag_start('text <uip:cite title="S" url="https://s.com" />')
            == -1
        )

    def test_partial_tag_at_start_of_string(self):
        assert _find_partial_tag_start("<uip:cite") == 0

    def test_partial_tag_just_cite_space(self):
        """'<uip:cite ' (with trailing space) is exactly the prefix, should buffer."""
        assert _find_partial_tag_start("hello <uip:cite ") == 6


class TestCitationStreamProcessor:
    def test_plain_text_passthrough(self):
        proc = CitationStreamProcessor()
        events = proc.add_chunk("Hello world")
        assert len(events) == 1
        assert events[0].data == "Hello world"
        assert events[0].citation is None

    def test_full_url_citation(self):
        proc = CitationStreamProcessor()
        text = 'A fact<uip:cite title="Doc" url="https://doc.com" />'
        events = proc.add_chunk(text)
        assert len(events) == 1
        assert events[0].data == "A fact"
        assert events[0].citation is not None
        assert events[0].citation.end is not None
        source = events[0].citation.end.sources[0]
        assert isinstance(source, UiPathConversationCitationSourceUrl)
        assert source.url == "https://doc.com"

    def test_full_reference_citation(self):
        proc = CitationStreamProcessor()
        text = 'A fact<uip:cite title="Report.pdf" reference="https://doc.com" page_number="3" />'
        events = proc.add_chunk(text)
        assert len(events) == 1
        assert events[0].citation is not None
        assert events[0].citation.end is not None
        source = events[0].citation.end.sources[0]
        assert isinstance(source, UiPathConversationCitationSourceMedia)
        assert source.download_url == "https://doc.com"
        assert source.page_number == "3"
        assert source.title == "Report.pdf"

    def test_buffering_partial_tag_then_completing(self):
        proc = CitationStreamProcessor()
        events1 = proc.add_chunk("Some text <uip:cite")
        assert len(events1) == 1
        assert events1[0].data == "Some text "

        events2 = proc.add_chunk(' title="S" url="https://s.com" />')
        assert len(events2) == 1
        assert events2[0].citation is not None

    def test_finalize_flushes_buffer(self):
        proc = CitationStreamProcessor()
        proc.add_chunk("text <uip:cite")
        events = proc.finalize()
        assert len(events) == 1
        assert events[0].data == "<uip:cite"
        assert events[0].citation is None

    def test_html_passthrough_for_coded_agents(self):
        proc = CitationStreamProcessor()
        events = proc.add_chunk("Use a <div> tag and <span> for styling.")
        assert len(events) == 1
        assert events[0].data == "Use a <div> tag and <span> for styling."
        assert events[0].citation is None

    def test_empty_text_input(self):
        """Empty string chunks produce no events."""
        proc = CitationStreamProcessor()
        events1 = proc.add_chunk("")
        assert events1 == []
        events2 = proc.add_chunk("")
        assert events2 == []
        events3 = proc.finalize()
        assert events3 == []

    def test_empty_chunks_interspersed(self):
        """Empty chunks between real text don't affect output."""
        proc = CitationStreamProcessor()
        all_events = []
        for chunk in ["Hello, ", "", "world!"]:
            all_events.extend(proc.add_chunk(chunk))
        all_events.extend(proc.finalize())
        combined = "".join(e.data for e in all_events if e.data)
        assert combined == "Hello, world!"
        assert all(e.citation is None for e in all_events)

    def test_multiple_citations_in_a_row(self):
        """Back-to-back citations: <uip:cite .../><uip:cite .../>."""
        proc = CitationStreamProcessor()
        text = (
            'First<uip:cite title="Doc1" url="https://doc1.com" />'
            '<uip:cite title="Doc2" url="https://doc2.com" />'
        )
        events = proc.add_chunk(text)
        events.extend(proc.finalize())

        cited_events = [e for e in events if e.citation is not None]
        assert len(cited_events) == 2
        assert cited_events[0].data == "First"
        assert cited_events[0].citation.end.sources[0].url == "https://doc1.com"
        # Second citation has empty text (back-to-back)
        assert cited_events[1].data == ""
        assert cited_events[1].citation.end.sources[0].url == "https://doc2.com"

    def test_citation_at_beginning_of_text(self):
        """Citation at the very start with no preceding text."""
        proc = CitationStreamProcessor()
        text = '<uip:cite title="Doc" url="https://doc.com" />Some text after.'
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        assert len(events) == 2
        assert events[0].data == ""
        assert events[0].citation is not None
        assert events[0].citation.end.sources[0].url == "https://doc.com"
        assert events[1].data == "Some text after."
        assert events[1].citation is None

    def test_partial_uip_something_else_emitted(self):
        """'<uip:something' does NOT match '<uip:cite ' prefix, so it's emitted immediately."""
        proc = CitationStreamProcessor()
        events = proc.add_chunk("text <uip:something else")
        events.extend(proc.finalize())
        combined = "".join(e.data for e in events if e.data)
        assert combined == "text <uip:something else"
        assert all(e.citation is None for e in events)

    def test_uip_prefix_followed_by_valid_citation(self):
        """'<uip' then '<uip:cite .../>' - the '<uip' is plain text, citation is parsed."""
        proc = CitationStreamProcessor()
        # First chunk: "<uip" gets buffered as potential partial tag
        events1 = proc.add_chunk("<uip")
        # "<uip" is a partial prefix of "<uip:cite ", so it should be buffered
        assert events1 == []

        # Second chunk completes with a space (not ":" so the buffer is not a valid prefix anymore)
        events2 = proc.add_chunk(
            ' <uip:cite title="Source" url="https://example.com" />'
        )
        events2.extend(proc.finalize())
        combined = "".join(e.data for e in events2 if e.data)
        cited = [e for e in events2 if e.citation is not None]
        # "<uip " is emitted as plain text, citation is parsed
        assert "<uip " in combined
        assert len(cited) == 1
        assert cited[0].citation.end.sources[0].url == "https://example.com"

    def test_streaming_citation_split_across_many_chunks(self):
        """Citation tag split character by character across chunks."""
        proc = CitationStreamProcessor()
        full_tag = '<uip:cite title="S" url="https://s.com" />'
        text = "Fact" + full_tag + " more"
        all_events = []
        # Feed one character at a time
        for ch in text:
            all_events.extend(proc.add_chunk(ch))
        all_events.extend(proc.finalize())

        combined = "".join(e.data for e in all_events if e.data)
        assert combined == "Fact more"
        cited = [e for e in all_events if e.citation is not None]
        assert len(cited) == 1
        # "Fact" was emitted as plain text before the '<' started buffering,
        # so the citation event has empty data (back-to-back with preceding text)
        assert cited[0].data == ""
        assert cited[0].citation.end.sources[0].url == "https://s.com"

    def test_multiple_citations_streamed_across_chunks(self):
        """Multiple citations arriving across chunk boundaries."""
        proc = CitationStreamProcessor()
        chunks = [
            'This is <uip:cite title="Doc1" url="https://doc1.co',
            'm" /> and this is <uip:cite title="Doc2" url="https://doc2.com',
            '" /> end',
        ]
        all_events = []
        for chunk in chunks:
            all_events.extend(proc.add_chunk(chunk))
        all_events.extend(proc.finalize())

        combined = "".join(e.data for e in all_events if e.data)
        assert combined == "This is  and this is  end"
        cited = [e for e in all_events if e.citation is not None]
        assert len(cited) == 2
        assert cited[0].citation.end.sources[0].url == "https://doc1.com"
        assert cited[1].citation.end.sources[0].url == "https://doc2.com"

    def test_finalize_flushes_unclosed_partial_tag_in_middle(self):
        """Partial tag mid-text: finalize emits buffered content as plain text."""
        proc = CitationStreamProcessor()
        events1 = proc.add_chunk("Text with<uip:cite source'")
        # "Text with" is not buffered since "<uip:cite source'" doesn't match prefix
        # (it has a quote character breaking the prefix pattern)
        events1.extend(proc.finalize())
        combined = "".join(e.data for e in events1 if e.data)
        assert combined == "Text with<uip:cite source'"
        assert all(e.citation is None for e in events1)

    def test_citation_deduplication_numbering(self):
        """Same citation source used twice gets the same number."""
        proc = CitationStreamProcessor()
        text = (
            'A<uip:cite title="Doc" url="https://doc.com" />'
            'B<uip:cite title="Doc" url="https://doc.com" />'
        )
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        cited = [e for e in events if e.citation is not None]
        assert len(cited) == 2
        # Same source should get the same number
        assert (
            cited[0].citation.end.sources[0].number
            == cited[1].citation.end.sources[0].number
        )

    def test_invalid_citation_skipped_in_stream(self):
        """Citation with neither url nor reference is skipped, text still emitted."""
        proc = CitationStreamProcessor()
        text = 'Some text<uip:cite title="Invalid" page_number="3" /> more text'
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        combined = "".join(e.data for e in events if e.data)
        assert combined == "Some text more text"
        assert all(e.citation is None for e in events)

    def test_mixed_valid_and_invalid_citations(self):
        """Only valid citations produce citation events, invalid ones are skipped."""
        proc = CitationStreamProcessor()
        text = (
            'A<uip:cite title="Valid" url="https://v.com" />'
            'B<uip:cite title="Invalid" page_number="1" />'
            'C<uip:cite title="Also Valid" reference="https://r.com" />'
        )
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        cited = [e for e in events if e.citation is not None]
        assert len(cited) == 2
        assert cited[0].citation.end.sources[0].url == "https://v.com"
        assert isinstance(
            cited[1].citation.end.sources[0], UiPathConversationCitationSourceMedia
        )
        assert cited[1].citation.end.sources[0].download_url == "https://r.com"

    def test_citation_at_end_of_text_streamed(self):
        """Citation at the end of text, tag split across chunks."""
        proc = CitationStreamProcessor()
        chunks = [
            'Here is some information<uip:cite title="Test Document" url="https://example.com',
            '"/>',
        ]
        all_events = []
        for chunk in chunks:
            all_events.extend(proc.add_chunk(chunk))
        all_events.extend(proc.finalize())

        combined = "".join(e.data for e in all_events if e.data)
        assert combined == "Here is some information"
        cited = [e for e in all_events if e.citation is not None]
        assert len(cited) == 1
        assert cited[0].citation.end.sources[0].title == "Test Document"
        assert cited[0].citation.end.sources[0].url == "https://example.com"

    def test_citation_in_middle_of_text_streamed(self):
        """Citation between text, tag split across chunks."""
        proc = CitationStreamProcessor()
        chunks = [
            'Here is some information<uip:cite title="Test Document" url="https://ex',
            'ample.com"/> and more text',
        ]
        all_events = []
        for chunk in chunks:
            all_events.extend(proc.add_chunk(chunk))
        all_events.extend(proc.finalize())

        combined = "".join(e.data for e in all_events if e.data)
        assert combined == "Here is some information and more text"
        cited = [e for e in all_events if e.citation is not None]
        assert len(cited) == 1
        assert cited[0].citation.end.sources[0].title == "Test Document"
        assert cited[0].citation.end.sources[0].url == "https://example.com"
        # Trailing text emitted without citation
        non_cited = [e for e in all_events if e.citation is None and e.data]
        assert any(" and more text" in e.data for e in non_cited)

    def test_multiple_citations_with_trailing_partial_tag(self):
        """Multiple citations across chunks with trailing partial tag at end."""
        proc = CitationStreamProcessor()
        chunks = [
            'This is <uip:cite title="Doc1" url="https://doc1.co',
            'm"/> and this is <uip:cite title="Doc2" reference="https://doc2.com" page_number="18',
            '"/>for you<uip:ci',
        ]
        all_events = []
        for chunk in chunks:
            all_events.extend(proc.add_chunk(chunk))
        all_events.extend(proc.finalize())

        combined = "".join(e.data for e in all_events if e.data)
        assert combined == "This is  and this is for you<uip:ci"
        cited = [e for e in all_events if e.citation is not None]
        assert len(cited) == 2
        # First citation - URL type
        assert isinstance(
            cited[0].citation.end.sources[0], UiPathConversationCitationSourceUrl
        )
        assert cited[0].citation.end.sources[0].url == "https://doc1.com"
        assert cited[0].citation.end.sources[0].title == "Doc1"
        # Second citation - Reference type with page number
        assert isinstance(
            cited[1].citation.end.sources[0], UiPathConversationCitationSourceMedia
        )
        assert cited[1].citation.end.sources[0].download_url == "https://doc2.com"
        assert cited[1].citation.end.sources[0].title == "Doc2"
        assert cited[1].citation.end.sources[0].page_number == "18"

    def test_finalize_with_no_prior_chunks(self):
        """Finalize with no add_chunk calls returns empty."""
        proc = CitationStreamProcessor()
        events = proc.finalize()
        assert events == []

    def test_just_open_bracket_at_end_of_chunk(self):
        """A lone '<' at end of chunk is buffered, then emitted on finalize."""
        proc = CitationStreamProcessor()
        events1 = proc.add_chunk("text <")
        assert len(events1) == 1
        assert events1[0].data == "text "
        events_final = proc.finalize()
        assert len(events_final) == 1
        assert events_final[0].data == "<"
        assert events_final[0].citation is None

    def test_just_open_bracket_followed_by_valid_citation(self):
        """A lone '<' buffered, next chunk completes a valid citation."""
        proc = CitationStreamProcessor()
        events1 = proc.add_chunk("text <")
        assert len(events1) == 1
        assert events1[0].data == "text "
        events2 = proc.add_chunk('uip:cite title="S" url="https://s.com" />')
        events2.extend(proc.finalize())
        cited = [e for e in events2 if e.citation is not None]
        assert len(cited) == 1
        assert cited[0].citation.end.sources[0].url == "https://s.com"

    def test_self_closing_tag_split_at_slash(self):
        """Self-closing /> split across chunks."""
        proc = CitationStreamProcessor()
        events1 = proc.add_chunk('Fact<uip:cite title="S" url="https://s.com" /')
        # The whole tag is buffered because it hasn't closed yet
        assert len(events1) == 1
        assert events1[0].data == "Fact"
        events2 = proc.add_chunk(">")
        events2.extend(proc.finalize())
        cited = [e for e in events2 if e.citation is not None]
        assert len(cited) == 1
        assert cited[0].citation.end.sources[0].url == "https://s.com"

    def test_multiple_citations_no_text_between(self):
        """Three citations with no text between any of them."""
        proc = CitationStreamProcessor()
        text = (
            '<uip:cite title="A" url="https://a.com" />'
            '<uip:cite title="B" url="https://b.com" />'
            '<uip:cite title="C" url="https://c.com" />'
        )
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        cited = [e for e in events if e.citation is not None]
        assert len(cited) == 3
        assert all(e.data == "" for e in cited)
        assert cited[0].citation.end.sources[0].url == "https://a.com"
        assert cited[1].citation.end.sources[0].url == "https://b.com"
        assert cited[2].citation.end.sources[0].url == "https://c.com"

    def test_citation_with_both_url_and_reference_skipped(self):
        """Citation with both url and reference is invalid and skipped."""
        proc = CitationStreamProcessor()
        text = 'Text<uip:cite title="Bad" url="https://u.com" reference="https://r.com" /> more'
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        combined = "".join(e.data for e in events if e.data)
        assert combined == "Text more"
        assert all(e.citation is None for e in events)

    def test_only_whitespace_between_citations(self):
        """Citations separated only by whitespace."""
        proc = CitationStreamProcessor()
        text = (
            '<uip:cite title="A" url="https://a.com" />'
            "   "
            '<uip:cite title="B" url="https://b.com" />'
        )
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        cited = [e for e in events if e.citation is not None]
        assert len(cited) == 2
        assert cited[0].data == ""
        assert cited[0].citation.end.sources[0].url == "https://a.com"
        assert cited[1].data == "   "
        assert cited[1].citation.end.sources[0].url == "https://b.com"

    def test_unclosed_tag_with_valid_content_before(self):
        """Valid citation followed by an unclosed partial tag at end of stream."""
        proc = CitationStreamProcessor()
        chunks = [
            'Info<uip:cite title="Doc" url="https://doc.com" /> trailing<uip:cite ',
        ]
        all_events = []
        for chunk in chunks:
            all_events.extend(proc.add_chunk(chunk))
        all_events.extend(proc.finalize())
        combined = "".join(e.data for e in all_events if e.data)
        assert combined == "Info trailing<uip:cite "
        cited = [e for e in all_events if e.citation is not None]
        assert len(cited) == 1
        assert cited[0].citation.end.sources[0].url == "https://doc.com"

    def test_angle_bracket_in_non_tag_context(self):
        """Angle brackets in math expressions don't trigger buffering."""
        proc = CitationStreamProcessor()
        events = proc.add_chunk("if x > 5 and y > 10")
        events.extend(proc.finalize())
        combined = "".join(e.data for e in events if e.data)
        assert combined == "if x > 5 and y > 10"
        assert all(e.citation is None for e in events)

    def test_finalize_idempotent(self):
        """Calling finalize multiple times is safe — second call returns empty."""
        proc = CitationStreamProcessor()
        # Use a partial tag so content is buffered for finalize
        proc.add_chunk("Hello <uip:cite")
        events1 = proc.finalize()
        assert len(events1) == 1
        assert events1[0].data == "<uip:cite"
        events2 = proc.finalize()
        assert events2 == []

    def test_add_chunk_after_finalize(self):
        """Parser can accept new chunks after finalize (reusable, unlike C# which throws)."""
        proc = CitationStreamProcessor()
        proc.add_chunk("Hello")
        proc.finalize()
        events = proc.add_chunk(" world")
        events.extend(proc.finalize())
        combined = "".join(e.data for e in events if e.data)
        assert combined == " world"
        assert all(e.citation is None for e in events)

    def test_add_chunk_after_finalize_preserves_source_numbering(self):
        """Source numbering continues after finalize — dedup state is preserved."""
        proc = CitationStreamProcessor()
        text1 = 'A<uip:cite title="Doc" url="https://doc.com" />'
        events1 = proc.add_chunk(text1)
        events1.extend(proc.finalize())
        cited1 = [e for e in events1 if e.citation is not None]
        assert cited1[0].citation.end.sources[0].number == 1

        # After finalize, same source keeps its number
        text2 = 'B<uip:cite title="Doc" url="https://doc.com" />'
        events2 = proc.add_chunk(text2)
        events2.extend(proc.finalize())
        cited2 = [e for e in events2 if e.citation is not None]
        assert cited2[0].citation.end.sources[0].number == 1

    def test_malformed_cite_colon_emitted_as_text(self):
        """'<uip:cite:' (colon instead of space after cite) is not a valid tag, emitted as plain text."""
        proc = CitationStreamProcessor()
        text = '<uip:cite: title="Source" url="https://example.com" />'
        events = proc.add_chunk(text)
        events.extend(proc.finalize())
        combined = "".join(e.data for e in events if e.data)
        assert combined == text
        assert all(e.citation is None for e in events)

    def test_uip_prefix_followed_by_citation_single_chunk(self):
        """'<uip ' followed by valid citation in a single chunk emits '<uip ' with the citation."""
        proc = CitationStreamProcessor()
        events = proc.add_chunk(
            '<uip <uip:cite title="Source" url="https://example.com" />'
        )
        events.extend(proc.finalize())
        cited = [e for e in events if e.citation is not None]
        assert len(cited) == 1
        assert cited[0].data == "<uip "
        assert cited[0].citation.end.sources[0].url == "https://example.com"


class TestExtractCitationsFromText:
    """Test cases for extract_citations_from_text function."""

    def test_no_citation_tags(self):
        """Text with no citation tags returns unchanged text and empty citations."""
        text = "This is plain text with no citations."
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == text
        assert citations == []

    def test_url_citation(self):
        """Text with a URL citation returns cleaned text and CitationData with URL source."""
        text = 'Some fact<uip:cite title="Doc" url="https://doc.com" />'
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == "Some fact"
        assert len(citations) == 1
        assert citations[0].offset == 0
        assert citations[0].length == 9  # len("Some fact")
        source = citations[0].sources[0]
        assert isinstance(source, UiPathConversationCitationSourceUrl)
        assert source.url == "https://doc.com"
        assert source.title == "Doc"
        assert source.number == 1

    def test_reference_media_citation(self):
        """Text with a reference citation returns CitationData with media source."""
        text = 'A finding<uip:cite title="Report.pdf" reference="https://r.com" page_number="5" />'
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == "A finding"
        assert len(citations) == 1
        source = citations[0].sources[0]
        assert isinstance(source, UiPathConversationCitationSourceMedia)
        assert source.download_url == "https://r.com"
        assert source.page_number == "5"
        assert source.title == "Report.pdf"

    def test_multiple_citations_correct_offsets(self):
        """Multiple citations on different spans get correct offsets and lengths."""
        text = (
            'First<uip:cite title="A" url="https://a.com" />'
            ' and second<uip:cite title="B" url="https://b.com" />'
        )
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == "First and second"
        assert len(citations) == 2
        # First citation: "First" at offset 0, length 5
        assert citations[0].offset == 0
        assert citations[0].length == 5
        assert citations[0].sources[0].url == "https://a.com"
        # Second citation: " and second" at offset 5, length 11
        assert citations[1].offset == 5
        assert citations[1].length == 11
        assert citations[1].sources[0].url == "https://b.com"

    def test_duplicate_sources_same_number(self):
        """Duplicate sources (same title+url) get the same number."""
        text = (
            'A<uip:cite title="Doc" url="https://doc.com" />'
            'B<uip:cite title="Doc" url="https://doc.com" />'
        )
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == "AB"
        assert len(citations) == 2
        assert citations[0].sources[0].number == citations[1].sources[0].number

    def test_back_to_back_citations_merged_into_previous(self):
        """Back-to-back citations merge the second source into the previous citation."""
        text = (
            'Text<uip:cite title="A" url="https://a.com" />'
            '<uip:cite title="B" url="https://b.com" />'
        )
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == "Text"
        assert len(citations) == 1
        assert len(citations[0].sources) == 2
        assert citations[0].sources[0].url == "https://a.com"
        assert citations[0].sources[1].url == "https://b.com"

    def test_three_back_to_back_citations(self):
        """Three back-to-back citations all merge into one citation with three sources."""
        text = (
            'Answer<uip:cite title="A" reference="https://a.com" page_number="1" />'
            '<uip:cite title="B" reference="https://b.com" page_number="2" />'
            '<uip:cite title="C" reference="https://c.com" page_number="3" />'
        )
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == "Answer"
        assert len(citations) == 1
        assert len(citations[0].sources) == 3
        assert citations[0].sources[0].title == "A"
        assert citations[0].sources[1].title == "B"
        assert citations[0].sources[2].title == "C"

    def test_back_to_back_citations_at_start_with_no_preceding_text(self):
        """Back-to-back citations at the very start (no preceding text) are all dropped."""
        text = (
            '<uip:cite title="A" url="https://a.com" />'
            '<uip:cite title="B" url="https://b.com" />'
        )
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == ""
        # First citation has no preceding text and no previous citation to merge into
        # Second citation also has no preceding text but merges into the first
        # Both end up dropped since neither has a text span
        assert len(citations) == 0

    def test_empty_text(self):
        """Empty string returns empty text and no citations."""
        cleaned, citations = extract_citations_from_text("")
        assert cleaned == ""
        assert citations == []

    def test_text_with_trailing_content(self):
        """Citation in middle of text, trailing text preserved."""
        text = 'A fact<uip:cite title="S" url="https://s.com" /> and more text.'
        cleaned, citations = extract_citations_from_text(text)
        assert cleaned == "A fact and more text."
        assert len(citations) == 1
        assert citations[0].offset == 0
        assert citations[0].length == 6  # len("A fact")

    def test_different_sources_get_different_numbers(self):
        """Different sources get incrementing numbers."""
        text = (
            'A<uip:cite title="Doc1" url="https://doc1.com" />'
            'B<uip:cite title="Doc2" url="https://doc2.com" />'
        )
        cleaned, citations = extract_citations_from_text(text)
        assert citations[0].sources[0].number == 1
        assert citations[1].sources[0].number == 2


class TestConvertCitationsToInlineTags:
    """Test cases for convert_citations_to_inline_tags function."""

    def test_basic_replacement(self):
        """Test basic [1] and [2] replacement with <uip:cite/> tags."""
        content = {
            "text": "Fact A [1] and fact B [2].",
            "citations": [
                {
                    "ordinal": 1,
                    "pageNumber": 3,
                    "source": "Report.pdf",
                    "reference": "https://example.com/ref1",
                },
                {
                    "ordinal": 2,
                    "pageNumber": 7,
                    "source": "Manual.pdf",
                    "reference": "https://example.com/ref2",
                },
            ],
        }

        result = convert_citations_to_inline_tags(content)

        assert "[1]" not in result
        assert "[2]" not in result
        assert (
            '<uip:cite title="Report.pdf" reference="https://example.com/ref1" page_number="3" />'
            in result
        )
        assert (
            '<uip:cite title="Manual.pdf" reference="https://example.com/ref2" page_number="7" />'
            in result
        )

    def test_same_ordinal_used_twice(self):
        """Test that the same ordinal appearing twice in text is replaced in both places."""
        content = {
            "text": "First mention [1] and second mention [1].",
            "citations": [
                {
                    "ordinal": 1,
                    "pageNumber": 5,
                    "source": "Doc.pdf",
                    "reference": "https://example.com/doc",
                },
            ],
        }

        result = convert_citations_to_inline_tags(content)

        assert "[1]" not in result
        assert result.count('<uip:cite title="Doc.pdf"') == 2

    def test_missing_citation_for_ordinal(self):
        """Test that ordinals without matching citations are left as-is."""
        content = {
            "text": "Has citation [1] and missing [3].",
            "citations": [
                {
                    "ordinal": 1,
                    "pageNumber": 1,
                    "source": "A.pdf",
                    "reference": "https://example.com/a",
                },
            ],
        }

        result = convert_citations_to_inline_tags(content)

        assert "[1]" not in result
        assert "[3]" in result
        assert '<uip:cite title="A.pdf"' in result

    def test_empty_citations_list(self):
        """Test that text is returned as-is when citations list is empty."""
        content = {
            "text": "No citations here [1].",
            "citations": [],
        }

        result = convert_citations_to_inline_tags(content)

        assert result == "No citations here [1]."

    def test_missing_text_field(self):
        """Test graceful handling when text field is missing."""
        content = {
            "citations": [
                {"ordinal": 1, "source": "A.pdf", "reference": "https://example.com"}
            ]
        }

        result = convert_citations_to_inline_tags(content)

        assert result == ""

    def test_missing_citations_field(self):
        """Test graceful handling when citations field is missing."""
        content = {"text": "Some text [1]."}

        result = convert_citations_to_inline_tags(content)

        assert result == "Some text [1]."

    def test_empty_content(self):
        """Test graceful handling of empty dict."""
        result = convert_citations_to_inline_tags({})

        assert result == ""

    def test_page_number_fallback_to_snake_case(self):
        """Test that page_number key is used as fallback when pageNumber is missing."""
        content = {
            "text": "Fact [1].",
            "citations": [
                {
                    "ordinal": 1,
                    "page_number": 10,
                    "source": "Guide.pdf",
                    "reference": "https://example.com/guide",
                },
            ],
        }

        result = convert_citations_to_inline_tags(content)

        assert 'page_number="10"' in result

    def test_missing_optional_fields_default_to_empty(self):
        """Test that missing source/reference default to empty strings."""
        content = {
            "text": "Fact [1].",
            "citations": [{"ordinal": 1}],
        }

        result = convert_citations_to_inline_tags(content)

        assert '<uip:cite title="" reference="" page_number="" />' in result

    def test_back_to_back_ordinals(self):
        """Test [1][2] back-to-back at end of text are both converted."""
        content = {
            "text": "Answer [1][2]",
            "citations": [
                {
                    "ordinal": 1,
                    "pageNumber": 1,
                    "source": "A.pdf",
                    "reference": "https://a.com",
                },
                {
                    "ordinal": 2,
                    "pageNumber": 5,
                    "source": "B.pdf",
                    "reference": "https://b.com",
                },
            ],
        }

        result = convert_citations_to_inline_tags(content)

        assert "[1]" not in result
        assert "[2]" not in result
        assert '<uip:cite title="A.pdf"' in result
        assert '<uip:cite title="B.pdf"' in result

    def test_back_to_back_ordinals_roundtrip_with_extract(self):
        """End-to-end: [1][2] converted to inline tags then extracted as merged sources."""
        content = {
            "text": "Answer [1][2]",
            "citations": [
                {
                    "ordinal": 1,
                    "pageNumber": 1,
                    "source": "A.pdf",
                    "reference": "https://a.com",
                },
                {
                    "ordinal": 2,
                    "pageNumber": 5,
                    "source": "B.pdf",
                    "reference": "https://b.com",
                },
            ],
        }

        inline_text = convert_citations_to_inline_tags(content)
        cleaned, citations = extract_citations_from_text(inline_text)

        assert cleaned == "Answer "
        assert len(citations) == 1
        assert len(citations[0].sources) == 2
        assert citations[0].sources[0].title == "A.pdf"
        assert citations[0].sources[1].title == "B.pdf"


class TestDeepRagCitationWrapper:
    """Test cases for cas_deep_rag_citation_wrapper function."""

    @pytest.mark.asyncio
    async def test_wrapper_transforms_result(self):
        """Test that the wrapper calls tool.ainvoke and transforms the JSON result."""
        raw_data = {
            "text": "Answer [1].",
            "citations": [
                {
                    "ordinal": 1,
                    "pageNumber": 3,
                    "source": "Report.pdf",
                    "reference": "https://example.com/ref",
                },
            ],
            "deepRagId": "task-123",
        }
        tool_message = ToolMessage(
            content=json.dumps(raw_data),
            name="test_tool",
            tool_call_id="call-1",
        )

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value=tool_message)
        mock_call = ToolCall(name="test_tool", args={}, id="call-1")

        result = await cas_deep_rag_citation_wrapper(mock_tool, mock_call)

        mock_tool.ainvoke.assert_called_once_with(mock_call)
        parsed = json.loads(result.content)
        assert "[1]" not in parsed["text"]
        assert '<uip:cite title="Report.pdf"' in parsed["text"]

    @pytest.mark.asyncio
    async def test_wrapper_replaces_all_ordinals(self):
        """Test that all ordinal references are replaced with inline cite tags."""
        raw_data = {
            "text": "Fact [1] and [2].",
            "citations": [
                {
                    "ordinal": 1,
                    "pageNumber": 1,
                    "source": "A.pdf",
                    "reference": "https://a.com",
                },
                {
                    "ordinal": 2,
                    "pageNumber": 2,
                    "source": "B.pdf",
                    "reference": "https://b.com",
                },
            ],
            "deepRagId": "task-456",
        }
        tool_message = ToolMessage(
            content=json.dumps(raw_data),
            name="test_tool",
            tool_call_id="call-2",
        )

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value=tool_message)
        mock_call = ToolCall(name="test_tool", args={}, id="call-2")

        result = await cas_deep_rag_citation_wrapper(mock_tool, mock_call)

        parsed = json.loads(result.content)
        assert "[1]" not in parsed["text"]
        assert "[2]" not in parsed["text"]
        assert '<uip:cite title="A.pdf"' in parsed["text"]
        assert '<uip:cite title="B.pdf"' in parsed["text"]

    @pytest.mark.asyncio
    async def test_wrapper_passthrough_on_invalid_json(self):
        """Test that the wrapper returns unmodified content when JSON parsing fails."""
        original_content = "not valid json"
        tool_message = ToolMessage(
            content=original_content,
            name="test_tool",
            tool_call_id="call-3",
        )

        mock_tool = AsyncMock()
        mock_tool.ainvoke = AsyncMock(return_value=tool_message)
        mock_call = ToolCall(name="test_tool", args={}, id="call-3")

        result = await cas_deep_rag_citation_wrapper(mock_tool, mock_call)

        assert result.content == original_content
        assert result.artifact is None
