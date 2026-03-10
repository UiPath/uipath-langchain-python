"""Tests for multimodal — file download, size limits, and content block creation."""

import base64

import httpx
import pytest
from pytest_httpx import HTTPXMock

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.multimodal.invoke import build_file_content_block
from uipath_langchain.agent.multimodal.types import MAX_FILE_SIZE_BYTES, FileInfo
from uipath_langchain.agent.multimodal.utils import (
    download_file_base64,
    encode_streamed_base64,
)

FILE_URL = "https://blob.storage.example.com/file.pdf"


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


class TestBuildFileContentBlock:
    """Tests for build_file_content_block — size limit enforced during download."""

    async def test_small_image_succeeds(self, httpx_mock: HTTPXMock) -> None:
        content = b"tiny image bytes"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="photo.png", mime_type="image/png")

        block = await build_file_content_block(file_info)

        assert block["type"] == "image"

    async def test_small_pdf_succeeds(self, httpx_mock: HTTPXMock) -> None:
        content = b"tiny pdf bytes"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="doc.pdf", mime_type="application/pdf")

        block = await build_file_content_block(file_info)

        assert block["type"] == "file"

    async def test_rejects_file_exceeding_default_limit(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Files larger than MAX_FILE_SIZE_BYTES are rejected."""
        oversized_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        httpx_mock.add_response(url=FILE_URL, content=oversized_content)
        file_info = FileInfo(url=FILE_URL, name="huge.pdf", mime_type="application/pdf")

        with pytest.raises(AgentRuntimeError, match="exceeds"):
            await build_file_content_block(file_info)

    async def test_rejects_file_exceeding_custom_limit(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """The max_size parameter is respected."""
        content = b"x" * 100
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="big.png", mime_type="image/png")

        with pytest.raises(AgentRuntimeError, match="exceeds"):
            await build_file_content_block(file_info, max_size=10)

    async def test_file_within_custom_limit_succeeds(
        self, httpx_mock: HTTPXMock
    ) -> None:
        content = b"abc"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="small.png", mime_type="image/png")

        block = await build_file_content_block(file_info, max_size=1000)

        assert block["type"] == "image"

    async def test_unsupported_mime_type_raises(self, httpx_mock: HTTPXMock) -> None:
        content = b"some data"
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(url=FILE_URL, name="data.csv", mime_type="text/csv")

        with pytest.raises(ValueError, match="Unsupported"):
            await build_file_content_block(file_info)

    async def test_error_includes_filename(self, httpx_mock: HTTPXMock) -> None:
        """AgentRuntimeError from download includes the filename for debuggability."""
        content = b"x" * 200
        httpx_mock.add_response(url=FILE_URL, content=content)
        file_info = FileInfo(
            url=FILE_URL, name="report.pdf", mime_type="application/pdf"
        )

        with pytest.raises(AgentRuntimeError, match="report.pdf"):
            await build_file_content_block(file_info, max_size=100)
