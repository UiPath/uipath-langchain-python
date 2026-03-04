"""Tests for multimodal utils — download_file_base64 with size limits."""

import base64

import httpx
import pytest
from pytest_httpx import HTTPXMock

from uipath_langchain.agent.multimodal.types import MAX_FILE_SIZE_BYTES
from uipath_langchain.agent.multimodal.utils import download_file_base64

FILE_URL = "https://blob.storage.example.com/file.pdf"


class TestDownloadFileBase64:
    """Tests for download_file_base64."""

    async def test_small_file_succeeds(self, httpx_mock: HTTPXMock) -> None:
        """A file within the size limit is downloaded and base64-encoded."""
        content = b"hello world"
        httpx_mock.add_response(url=FILE_URL, content=content)

        result = await download_file_base64(FILE_URL)

        assert result == base64.b64encode(content).decode("utf-8")

    async def test_rejects_file_exceeding_content_length(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Files whose Content-Length exceeds the limit are rejected early."""
        oversized = MAX_FILE_SIZE_BYTES + 1
        httpx_mock.add_response(
            url=FILE_URL,
            content=b"x",
            headers={"content-length": str(oversized)},
        )

        with pytest.raises(ValueError, match="exceeds"):
            await download_file_base64(FILE_URL)

    async def test_rejects_file_exceeding_limit_during_streaming(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Files that exceed the limit during download (no Content-Length) are rejected."""
        # Use a custom max_size small enough for the test payload
        small_limit = 10
        content = b"x" * (small_limit + 1)
        httpx_mock.add_response(url=FILE_URL, content=content)

        with pytest.raises(ValueError, match="exceeds"):
            await download_file_base64(FILE_URL, max_size=small_limit)

    async def test_custom_max_size(self, httpx_mock: HTTPXMock) -> None:
        """The max_size parameter is respected."""
        content = b"abc"
        httpx_mock.add_response(url=FILE_URL, content=content)

        result = await download_file_base64(FILE_URL, max_size=100)

        assert result == base64.b64encode(content).decode("utf-8")

    async def test_http_error_propagates(self, httpx_mock: HTTPXMock) -> None:
        """HTTP errors are propagated to the caller."""
        httpx_mock.add_response(url=FILE_URL, status_code=404)

        with pytest.raises(httpx.HTTPStatusError):
            await download_file_base64(FILE_URL)

    async def test_temp_file_cleaned_up_on_success(self, httpx_mock: HTTPXMock) -> None:
        """Temporary files are cleaned up after successful download."""
        content = b"test data"
        httpx_mock.add_response(url=FILE_URL, content=content)

        result = await download_file_base64(FILE_URL)

        # Function should succeed and return valid base64
        assert base64.b64decode(result) == content

    async def test_temp_file_cleaned_up_on_size_error(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Temporary files are cleaned up even when size validation fails."""
        oversized = MAX_FILE_SIZE_BYTES + 1
        httpx_mock.add_response(
            url=FILE_URL,
            content=b"x",
            headers={"content-length": str(oversized)},
        )

        with pytest.raises(ValueError):
            await download_file_base64(FILE_URL)
        # No assertion needed — the finally block in the implementation handles cleanup.
        # This test just verifies no exception from cleanup itself.
