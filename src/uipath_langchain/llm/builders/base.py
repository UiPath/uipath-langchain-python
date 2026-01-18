"""Abstract base class for message content builders."""

import base64
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

IMAGE_MIME_TYPES: set[str] = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}


def is_pdf(mime_type: str) -> bool:
    """Check if the MIME type represents a PDF document."""
    return mime_type.lower() == "application/pdf"


def is_image(mime_type: str) -> bool:
    """Check if the MIME type represents a supported image format."""
    return mime_type.lower() in IMAGE_MIME_TYPES


def sanitize_filename_for_anthropic(filename: str) -> str:
    """Sanitize a filename to conform to Anthropic's document naming requirements."""
    if not filename or filename.isspace():
        return "document"

    sanitized = re.sub(r"[^a-zA-Z0-9_\s\-\(\)\[\]\.]", "_", filename)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = sanitized.strip()

    return sanitized if sanitized else "document"


async def download_file_bytes(url: str) -> bytes:
    """Download a file from a URL and return its content bytes."""
    async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def download_file_base64(url: str) -> str:
    """Download a file from a URL and return its content as a base64 string."""
    file_content = await download_file_bytes(url)
    return base64.b64encode(file_content).decode("utf-8")


class MessageContentBuilder(ABC):
    """Abstract base class for building provider-specific message content parts."""

    @abstractmethod
    async def build_file_content_part(
        self,
        url: str,
        filename: str,
        mime_type: str,
    ) -> dict[str, Any]:
        """Build a provider-specific content part for a file.

        Args:
            url: URL to download the file from.
            filename: Name of the file.
            mime_type: MIME type of the file.

        Returns:
            Provider-specific content part dictionary.
        """
