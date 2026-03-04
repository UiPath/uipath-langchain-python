"""Utility functions for multimodal file handling."""

import base64
import logging
import re
import tempfile
from pathlib import Path

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

from .types import IMAGE_MIME_TYPES, MAX_FILE_SIZE_BYTES

logger = logging.getLogger("uipath")


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


async def download_file_base64(
    url: str,
    *,
    max_size: int = MAX_FILE_SIZE_BYTES,
) -> str:
    """Download a file from a URL and return its content as a base64 string.

    Uses streaming download to validate file size before loading into memory
    and writes to a temporary file to avoid holding raw bytes and the encoded
    string simultaneously, reducing peak memory usage.

    Args:
        url: The URL to download the file from.
        max_size: Maximum allowed file size in bytes.

    Raises:
        ValueError: If the file exceeds the maximum allowed size.
    """
    tmp_path: str | None = None
    try:
        async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                # Early reject based on Content-Length header
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_size:
                    raise ValueError(
                        f"File size {int(content_length)} bytes exceeds"
                        f" the {max_size} byte limit"
                    )

                # Stream to a temporary file to keep raw bytes off the heap
                total_bytes = 0
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = tmp.name
                    async for chunk in response.aiter_bytes(chunk_size=65_536):
                        total_bytes += len(chunk)
                        if total_bytes > max_size:
                            raise ValueError(
                                f"File size exceeds the {max_size} byte"
                                " limit (detected while downloading)"
                            )
                        tmp.write(chunk)

        logger.info("Downloaded file: %d bytes", total_bytes)

        # Read from disk and encode — free each intermediate before the next
        raw = Path(tmp_path).read_bytes()
        encoded = base64.b64encode(raw)
        del raw  # free raw bytes before creating the str copy
        result = encoded.decode("utf-8")
        del encoded  # free encoded bytes before finally block runs
        return result

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
