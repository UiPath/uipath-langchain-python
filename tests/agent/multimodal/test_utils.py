"""Tests for multimodal — file download, size limits, and content block creation."""

import base64
import io

import httpx
import pytest
from PIL import Image
from pytest_httpx import HTTPXMock

from uipath_langchain.agent.multimodal.invoke import build_file_content_blocks_for
from uipath_langchain.agent.multimodal.types import MAX_FILE_SIZE_BYTES, FileInfo
from uipath_langchain.agent.multimodal.utils import (
    download_file_base64,
    encode_streamed_base64,
    normalize_mime_type,
)

FILE_URL = "https://blob.storage.example.com/file.pdf"


def _make_tiff(num_pages: int = 1, width: int = 2, height: int = 2) -> bytes:
    """Create a minimal valid TIFF image with the given number of pages."""
    frames = [
        Image.new("RGBA", (width, height), color=(i * 40, 0, 0, 255))
        for i in range(num_pages)
    ]
    buf = io.BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


async def _async_iter(chunks: list[bytes]):
    """Helper: wrap a list of byte chunks as an async iterator."""
    for chunk in chunks:
        yield chunk


class TestEncodeStreamedBase64:
    """Tests for encode_streamed_base64 — incremental encoding with size limit."""

    async def test_encodes_single_chunk(self) -> None:
        content = b"hello world"
        result = await encode_streamed_base64(_async_iter([content]))
        assert result == base64.b64encode(content).decode("ascii")

    async def test_encodes_multiple_chunks(self) -> None:
        chunks = [b"hello ", b"world"]
        result = await encode_streamed_base64(_async_iter(chunks))
        assert result == base64.b64encode(b"hello world").decode("ascii")

    async def test_encodes_empty_stream(self) -> None:
        result = await encode_streamed_base64(_async_iter([]))
        assert result == ""

    async def test_encodes_single_byte_chunks(self) -> None:
        """Handles worst-case chunking where every chunk is 1 byte."""
        data = b"abcdefgh"
        chunks = [bytes([b]) for b in data]
        result = await encode_streamed_base64(_async_iter(chunks))
        assert result == base64.b64encode(data).decode("ascii")

    async def test_rejects_when_exceeds_max_size(self) -> None:
        chunks = [b"x" * 60, b"x" * 60]
        with pytest.raises(ValueError, match="exceeds"):
            await encode_streamed_base64(_async_iter(chunks), max_size=100)

    async def test_allows_exactly_at_limit(self) -> None:
        content = b"x" * 100
        result = await encode_streamed_base64(_async_iter([content]), max_size=100)
        assert result == base64.b64encode(content).decode("ascii")

    async def test_unlimited_when_max_size_zero(self) -> None:
        content = b"x" * 10_000
        result = await encode_streamed_base64(_async_iter([content]), max_size=0)
        assert result == base64.b64encode(content).decode("ascii")

    async def test_error_message_formats_as_mb(self) -> None:
        """Error message shows MB, not raw bytes."""
        limit = 10 * 1024 * 1024  # 10 MB
        chunks = [b"x" * (limit + 1)]
        with pytest.raises(ValueError, match=r"10 MB.*limit"):
            await encode_streamed_base64(_async_iter(chunks), max_size=limit)


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class TestNormalizeMimeType:
    """Tests for normalize_mime_type — the filename extension repairs a wrong label."""

    def test_csv_declared_as_excel_becomes_text_csv(self) -> None:
        """The prod failure: Windows labels .csv as application/vnd.ms-excel and
        OpenAI then parses the text as a corrupt binary workbook."""
        result = normalize_mime_type(
            "application/vnd.ms-excel", "Factset_Output_20260923.csv"
        )
        assert result == "text/csv"

    def test_matching_type_is_unchanged(self) -> None:
        assert normalize_mime_type("text/csv", "data.csv") == "text/csv"
        assert normalize_mime_type("application/pdf", "doc.pdf") == "application/pdf"
        assert normalize_mime_type(XLSX_MIME, "EOD Pricing_EM.xlsx") == XLSX_MIME

    def test_real_xls_keeps_excel_type(self) -> None:
        result = normalize_mime_type("application/vnd.ms-excel", "legacy.xls")
        assert result == "application/vnd.ms-excel"

    def test_accepted_alias_is_unchanged(self) -> None:
        assert normalize_mime_type("application/csv", "data.csv") == "application/csv"
        assert normalize_mime_type("image/jpg", "photo.jpg") == "image/jpg"
        assert normalize_mime_type("text/xml", "feed.xml") == "text/xml"

    def test_parameters_do_not_trigger_a_rewrite(self) -> None:
        declared = "text/csv; charset=utf-8"
        assert normalize_mime_type(declared, "data.csv") == declared

    def test_comparison_is_case_insensitive(self) -> None:
        assert normalize_mime_type("Text/CSV", "DATA.CSV") == "Text/CSV"
        assert (
            normalize_mime_type("Application/Vnd.MS-Excel", "Report.CSV") == "text/csv"
        )

    def test_xlsx_declared_as_legacy_excel_becomes_xlsx(self) -> None:
        result = normalize_mime_type("application/vnd.ms-excel", "book.xlsx")
        assert result == XLSX_MIME

    def test_generic_octet_stream_is_replaced_for_known_extension(self) -> None:
        result = normalize_mime_type("application/octet-stream", "scan.png")
        assert result == "image/png"

    def test_missing_type_is_derived_from_known_extension(self) -> None:
        assert normalize_mime_type("", "report.pdf") == "application/pdf"
        assert normalize_mime_type("", "notes.txt") == "text/plain"

    def test_unknown_extension_keeps_declared_type(self) -> None:
        declared = "application/octet-stream"
        assert normalize_mime_type(declared, "blob.bin") == declared
        assert normalize_mime_type("", "blob.bin") == ""

    def test_no_extension_keeps_declared_type(self) -> None:
        assert normalize_mime_type("text/plain", "README") == "text/plain"
        assert normalize_mime_type("text/plain", "trailing.") == "text/plain"
        assert normalize_mime_type("text/plain", "") == "text/plain"

    def test_only_the_last_suffix_counts(self) -> None:
        assert (
            normalize_mime_type("application/gzip", "dump.csv.gz") == "application/gzip"
        )
        assert normalize_mime_type("application/x-tar", "data.tar.csv") == "text/csv"

    def test_masked_filename_keeps_its_extension(self) -> None:
        """PII-masked copies are renamed with a prefix and must normalize the same."""
        result = normalize_mime_type(
            "application/vnd.ms-excel", "pii_masked_Factset_Output_20260923.csv"
        )
        assert result == "text/csv"


class TestDownloadFileBase64:
    """Tests for download_file_base64 — streaming download with optional size limit."""

    async def test_downloads_and_encodes(self, httpx_mock: HTTPXMock) -> None:
        content = b"hello world"
        httpx_mock.add_response(url=FILE_URL, content=content)

        result = await download_file_base64(FILE_URL)

        assert result == base64.b64encode(content).decode("utf-8")

    async def test_http_error_propagates(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=FILE_URL, status_code=404)

        with pytest.raises(httpx.HTTPStatusError):
            await download_file_base64(FILE_URL)

    async def test_rejects_via_content_length(self, httpx_mock: HTTPXMock) -> None:
        """Content-Length header triggers fast rejection before download."""
        oversized = b"x" * 200
        httpx_mock.add_response(url=FILE_URL, content=oversized)

        with pytest.raises(ValueError, match="exceeds"):
            await download_file_base64(FILE_URL, max_size=100)

    async def test_rejects_during_streaming(self, httpx_mock: HTTPXMock) -> None:
        """Streaming guard catches oversized files without Content-Length."""
        stream = _ChunkedStream([b"x" * 60, b"x" * 60])
        httpx_mock.add_response(
            url=FILE_URL,
            stream=stream,
            headers={"transfer-encoding": "chunked"},
        )

        with pytest.raises(ValueError, match="exceeds"):
            await download_file_base64(FILE_URL, max_size=100)

    async def test_invalid_content_length_falls_back_to_streaming(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Malformed Content-Length should not crash download logic."""
        content = b"x" * 80
        httpx_mock.add_response(
            url=FILE_URL,
            content=content,
            headers={"content-length": "invalid"},
        )

        result = await download_file_base64(FILE_URL, max_size=100)

        assert result == base64.b64encode(content).decode("utf-8")

    async def test_unlimited_when_max_size_zero(self, httpx_mock: HTTPXMock) -> None:
        """Default max_size=0 allows any file size."""
        content = b"x" * 10_000
        httpx_mock.add_response(url=FILE_URL, content=content)

        result = await download_file_base64(FILE_URL, max_size=0)

        assert result == base64.b64encode(content).decode("utf-8")

    async def test_file_exactly_at_limit_succeeds(self, httpx_mock: HTTPXMock) -> None:
        content = b"x" * 100
        httpx_mock.add_response(url=FILE_URL, content=content)

        result = await download_file_base64(FILE_URL, max_size=100)

        assert result == base64.b64encode(content).decode("utf-8")


class TestBuildFileContentBlocksFor:
    """Tests for build_file_content_blocks_for — size limit enforced during download."""

    async def test_small_image_succeeds(self, httpx_mock: HTTPXMock) -> None:
        content = b"tiny image bytes"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="photo.png", mime_type="image/png")

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"

    async def test_small_pdf_succeeds(self, httpx_mock: HTTPXMock) -> None:
        content = b"tiny pdf bytes"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="doc.pdf", mime_type="application/pdf")

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "file"

    async def test_rejects_file_exceeding_default_limit(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Files larger than MAX_FILE_SIZE_BYTES are rejected."""
        oversized_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        httpx_mock.add_response(url=FILE_URL, content=oversized_content)
        file_info = FileInfo(url=FILE_URL, name="huge.pdf", mime_type="application/pdf")

        with pytest.raises(ValueError, match="exceeds"):
            await build_file_content_blocks_for(file_info)

    async def test_rejects_file_exceeding_custom_limit(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """The max_size parameter is respected."""
        content = b"x" * 100
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="big.png", mime_type="image/png")

        with pytest.raises(ValueError, match="exceeds"):
            await build_file_content_blocks_for(file_info, max_size=10)

    async def test_file_within_custom_limit_succeeds(
        self, httpx_mock: HTTPXMock
    ) -> None:
        content = b"abc"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="small.png", mime_type="image/png")

        blocks = await build_file_content_blocks_for(file_info, max_size=1000)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"

    async def test_arbitrary_mime_type_returns_file_block(
        self, httpx_mock: HTTPXMock
    ) -> None:
        content = b"col1,col2\n1,2\n"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="data.csv", mime_type="text/csv")

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "file"
        assert blocks[0]["mime_type"] == "text/csv"

    async def test_unsupported_mime_type_returns_file_block(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Format support is delegated to the LLM/provider (#842): an arbitrary
        MIME type is wrapped in a file block here, not rejected. A provider that
        cannot read it raises at the model-invocation boundary, where it is
        translated into a USER error."""
        content = b"\x00\x01\x02"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(
            url=FILE_URL, name="blob.bin", mime_type="application/octet-stream"
        )

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "file"
        assert blocks[0]["mime_type"] == "application/octet-stream"

    async def test_empty_mime_type_defaults_to_octet_stream(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """An attachment with no MIME type defaults to octet-stream and is passed
        through as a file block (delegated to the LLM)."""
        content = b"data"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="blob.bin", mime_type="")

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "file"
        assert blocks[0]["mime_type"] == "application/octet-stream"

    async def test_error_includes_filename(self, httpx_mock: HTTPXMock) -> None:
        """ValueError from download includes the filename for debuggability."""
        content = b"x" * 200
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(
            url=FILE_URL, name="report.pdf", mime_type="application/pdf"
        )

        with pytest.raises(ValueError, match="report.pdf"):
            await build_file_content_blocks_for(file_info, max_size=100)

    async def test_csv_declared_as_excel_is_sent_as_text_csv(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Regression: a Windows uploader labels .csv as application/vnd.ms-excel.
        With the sanitized filename carrying no extension, OpenAI parsed the CSV
        text as a binary workbook and rejected it as corrupted (invalid_file).
        The block must carry text/csv while the filename stays sanitized."""
        content = b"Ticker,Price\r\nEM001,11.37\r\n"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(
            url=FILE_URL,
            name="Factset_Output_20260923.csv",
            mime_type="application/vnd.ms-excel",
        )

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "file"
        assert blocks[0]["mime_type"] == "text/csv"
        assert blocks[0]["extras"]["filename"] == "Factset-Output-20260923-csv"
        assert base64.b64decode(blocks[0]["base64"]) == content

    async def test_image_declared_as_octet_stream_becomes_image_block(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """The normalized type also drives the image/file block decision."""
        content = b"tiny image bytes"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(
            url=FILE_URL, name="photo.png", mime_type="application/octet-stream"
        )

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["mime_type"] == "image/png"

    async def test_tiff_declared_as_octet_stream_is_split_into_pages(
        self, httpx_mock: HTTPXMock
    ) -> None:
        tiff_bytes = _make_tiff(num_pages=2)
        httpx_mock.add_response(url=FILE_URL, content=tiff_bytes)
        file_info = FileInfo(
            url=FILE_URL, name="scan.tiff", mime_type="application/octet-stream"
        )

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 2
        assert all(block["type"] == "image" for block in blocks)

    async def test_correctly_labeled_file_is_unchanged(
        self, httpx_mock: HTTPXMock
    ) -> None:
        content = b"%PDF-1.4 tiny"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(
            url=FILE_URL, name="EOD Pricing_EM.pdf", mime_type="application/pdf"
        )

        blocks = await build_file_content_blocks_for(file_info)

        assert blocks[0]["mime_type"] == "application/pdf"
        assert blocks[0]["extras"]["filename"] == "EOD Pricing-EM-pdf"

    async def test_single_page_tiff_returns_one_png_block(
        self, httpx_mock: HTTPXMock
    ) -> None:
        tiff_bytes = _make_tiff(num_pages=1)
        httpx_mock.add_response(url=FILE_URL, content=tiff_bytes)
        file_info = FileInfo(url=FILE_URL, name="scan.tiff", mime_type="image/tiff")

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["mime_type"] == "image/png"

    async def test_multi_page_tiff_returns_one_block_per_page(
        self, httpx_mock: HTTPXMock
    ) -> None:
        tiff_bytes = _make_tiff(num_pages=3)
        httpx_mock.add_response(url=FILE_URL, content=tiff_bytes)
        file_info = FileInfo(url=FILE_URL, name="doc.tiff", mime_type="image/tiff")

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 3
        for block in blocks:
            assert block["type"] == "image"
            assert block["mime_type"] == "image/png"

    async def test_tiff_x_tiff_mime_type_accepted(self, httpx_mock: HTTPXMock) -> None:
        tiff_bytes = _make_tiff(num_pages=1)
        httpx_mock.add_response(url=FILE_URL, content=tiff_bytes)
        file_info = FileInfo(url=FILE_URL, name="scan.tif", mime_type="image/x-tiff")

        blocks = await build_file_content_blocks_for(file_info)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"

    async def test_tiff_rejects_exceeding_size_limit(
        self, httpx_mock: HTTPXMock
    ) -> None:
        tiff_bytes = _make_tiff(num_pages=1)
        httpx_mock.add_response(url=FILE_URL, content=tiff_bytes)
        file_info = FileInfo(url=FILE_URL, name="big.tiff", mime_type="image/tiff")

        with pytest.raises(ValueError, match="exceeds"):
            await build_file_content_blocks_for(file_info, max_size=10)

    async def test_tiff_error_includes_filename(self, httpx_mock: HTTPXMock) -> None:
        tiff_bytes = _make_tiff(num_pages=1)
        httpx_mock.add_response(url=FILE_URL, content=tiff_bytes)
        file_info = FileInfo(url=FILE_URL, name="report.tiff", mime_type="image/tiff")

        with pytest.raises(ValueError, match="report.tiff"):
            await build_file_content_blocks_for(file_info, max_size=10)

    async def test_tiff_png_blocks_contain_valid_base64(
        self, httpx_mock: HTTPXMock
    ) -> None:
        tiff_bytes = _make_tiff(num_pages=1)
        httpx_mock.add_response(url=FILE_URL, content=tiff_bytes)
        file_info = FileInfo(url=FILE_URL, name="scan.tiff", mime_type="image/tiff")

        blocks = await build_file_content_blocks_for(file_info)

        png_bytes = base64.b64decode(blocks[0]["base64"])
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"
