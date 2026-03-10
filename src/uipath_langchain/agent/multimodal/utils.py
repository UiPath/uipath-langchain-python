"""Utility functions for multimodal file handling."""

import base64
import re
from collections.abc import AsyncIterator

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

from .types import IMAGE_MIME_TYPES


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to conform to provider document naming requirements.

    Bedrock only allows: alphanumeric characters, whitespace, hyphens,
    parentheses, and square brackets. No consecutive whitespace allowed.
    """
    if not filename or filename.isspace():
        return "document"

    sanitized = re.sub(r"[^a-zA-Z0-9\s\-\(\)\[\]]", "-", filename)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = re.sub(r"-+", "-", sanitized)
    sanitized = sanitized.strip(" -")

    return sanitized if sanitized else "document"


def is_pdf(mime_type: str) -> bool:
    """Check if the MIME type represents a PDF document."""
    return mime_type.lower() == "application/pdf"


def is_image(mime_type: str) -> bool:
    """Check if the MIME type represents a supported image format."""
    return mime_type.lower() in IMAGE_MIME_TYPES


def _format_mb(size_bytes: int, decimals: int = 1) -> str:
    """Format a byte count as MB.

    Args:
        size_bytes: Size in bytes.
        decimals: Number of decimal places (0 for rounded integer).
    """
    return f"{size_bytes / (1024 * 1024):.{decimals}f} MB"


async def encode_streamed_base64(
    chunks: AsyncIterator[bytes],
    *,
    max_size: int = 0,
) -> str:
    """Incrementally base64-encode an async stream of byte chunks.

    Encodes chunks as they arrive so the raw file bytes are never assembled
    into a single contiguous buffer. base64 processes 3-byte groups, so a
    remainder of 0-2 bytes is buffered between chunks.

    Args:
        chunks: Async iterator yielding raw byte chunks.
        max_size: Maximum allowed total size in bytes. 0 means unlimited.

    Returns:
        The full base64-encoded string.

    Raises:
        ValueError: If the total size exceeds max_size.
    """
    encoded_buf = bytearray()
    remainder = b""
    total = 0

    async for chunk in chunks:
        total += len(chunk)
        if max_size > 0 and total > max_size:
            raise ValueError(
                f"File exceeds the {_format_mb(max_size, decimals=0)}"
                f" limit for LLM payloads"
                f" (downloaded {_format_mb(total)} so far)"
            )

        data = remainder + chunk
        usable = len(data) - (len(data) % 3)
        if usable > 0:
            encoded_buf += base64.b64encode(data[:usable])
            remainder = data[usable:]
        else:
            remainder = data

    if remainder:
        encoded_buf += base64.b64encode(remainder)

    result = encoded_buf.decode("ascii")
    del encoded_buf
    return result


async def download_file_base64(url: str, *, max_size: int = 0) -> str:
    """Download a file from a URL and return its content as a base64 string.

    Args:
        url: The URL to download from.
        max_size: Maximum allowed file size in bytes. 0 means unlimited.

    Raises:
        ValueError: If the file exceeds max_size.
        httpx.HTTPStatusError: If the HTTP request fails.
    """
    async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            # Fast reject via Content-Length before reading the body
            if max_size > 0:
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        content_length_value = int(content_length)
                    except ValueError:
                        content_length_value = None
                    if (
                        content_length_value is not None
                        and content_length_value > max_size
                    ):
                        raise ValueError(
                            f"File is {_format_mb(content_length_value)}"
                            f" which exceeds the {_format_mb(max_size, decimals=0)}"
                            f" limit for Agent LLM payloads"
                        )

            return await encode_streamed_base64(
                response.aiter_bytes(), max_size=max_size
            )
