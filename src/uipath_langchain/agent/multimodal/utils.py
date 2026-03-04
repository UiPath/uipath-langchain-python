"""Utility functions for multimodal file handling."""

import base64
import re

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


async def download_file_base64(url: str, *, max_size: int = 0) -> str:
    """Download a file from a URL and return its content as a base64 string.

    Base64-encodes chunks incrementally during download so the raw file bytes
    are never assembled into a single contiguous buffer.

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
                            f"File is {content_length_value / (1024 * 1024):.1f} MB"
                            f" which exceeds the {max_size / (1024 * 1024):.0f} MB"
                            f" limit for Agent LLM payloads"
                        )

            # Encode base64 incrementally so raw bytes are never fully
            # assembled. base64 processes 3-byte groups, so we buffer a
            # remainder of 0-2 bytes between chunks.
            encoded_buf = bytearray()
            remainder = b""
            total = 0

            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if max_size > 0 and total > max_size:
                    raise ValueError(
                        f"File exceeds the {max_size / (1024 * 1024):.0f} MB"
                        f" limit for LLM payloads"
                        f" (downloaded {total / (1024 * 1024):.1f} MB so far)"
                    )

                data = remainder + chunk
                # Encode complete 3-byte groups; keep leftover for next chunk
                usable = len(data) - (len(data) % 3)
                if usable > 0:
                    encoded_buf += base64.b64encode(data[:usable])
                    remainder = data[usable:]
                else:
                    remainder = data

            # Encode any remaining bytes (with base64 padding)
            if remainder:
                encoded_buf += base64.b64encode(remainder)

    result = encoded_buf.decode("ascii")
    del encoded_buf
    return result
