"""Utility functions for multimodal file handling."""

import base64
import re

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

from .types import IMAGE_MIME_TYPES, TEXT_MIME_TYPES


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


def is_text(mime_type: str) -> bool:
    """Check if the MIME type represents a supported text format."""
    return mime_type.lower() in TEXT_MIME_TYPES


async def download_file_base64(url: str) -> str:
    """Download a file from a URL and return its content as a base64 string."""
    async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
        response = await client.get(url)
        response.raise_for_status()
        file_content = response.content
    return base64.b64encode(file_content).decode("utf-8")


async def download_file_text(url: str) -> str:
    """Download a file from a URL and return its content as a text string."""
    async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
