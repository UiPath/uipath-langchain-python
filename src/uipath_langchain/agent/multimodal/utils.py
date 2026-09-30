"""Utility functions for multimodal file handling."""

import base64
import io
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from langchain_core.messages import DataContentBlock
from uipath._utils._ssl_context import get_httpx_client_kwargs

from .types import IMAGE_MIME_TYPES, TIFF_MIME_TYPES


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


_XLS_MIME = "application/vnd.ms-excel"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# Canonical MIME type per file extension (first entry), followed by aliases that
# are equally acceptable. Curated on purpose: ``mimetypes.guess_type`` consults
# the OS registry on Windows and reports ``.csv`` as ``application/vnd.ms-excel``
# when Excel is installed, which is exactly the mislabel this table repairs.
_EXTENSION_MIME_TYPES: dict[str, tuple[str, ...]] = {
    # Spreadsheets and delimited text
    "csv": ("text/csv", "application/csv"),
    "tsv": ("text/tab-separated-values", "text/tsv"),
    "xls": (_XLS_MIME,),
    "xlsx": (_XLSX_MIME,),
    # Documents
    "pdf": ("application/pdf",),
    "doc": ("application/msword",),
    "docx": (_DOCX_MIME,),
    "ppt": ("application/vnd.ms-powerpoint",),
    "pptx": (_PPTX_MIME,),
    "rtf": ("application/rtf", "text/rtf"),
    # Text
    "txt": ("text/plain",),
    "md": ("text/markdown", "text/x-markdown"),
    "json": ("application/json",),
    "xml": ("application/xml", "text/xml"),
    "html": ("text/html",),
    "htm": ("text/html",),
    # Images
    "png": ("image/png",),
    "jpg": ("image/jpeg", "image/jpg"),
    "jpeg": ("image/jpeg", "image/jpg"),
    "gif": ("image/gif",),
    "webp": ("image/webp",),
    "tif": ("image/tiff", "image/x-tiff"),
    "tiff": ("image/tiff", "image/x-tiff"),
}


def _file_extension(filename: str) -> str:
    """Return the lower-cased extension of ``filename`` without the dot, or ``""``."""
    name = (filename or "").strip()
    if "." not in name:
        return ""
    ext = name.rsplit(".", 1)[1].strip().lower()
    return ext if ext.isalnum() else ""


def normalize_mime_type(mime_type: str, filename: str) -> str:
    """Return the MIME type to send to the provider for ``filename``.

    The declared type comes from whoever uploaded the attachment and is not
    always right: Windows clients label ``.csv`` as ``application/vnd.ms-excel``
    (the registry mapping when Excel is installed), others send
    ``application/octet-stream`` for everything, and some send nothing at all.
    Providers trust the MIME type — OpenAI reads it from the data URL and
    Bedrock maps it to the document ``format`` — so a mislabeled CSV is parsed
    as a legacy binary workbook and rejected as corrupted. The sanitized
    filename cannot compensate because it no longer carries an extension.

    When the filename has a known extension and the declared type is missing,
    generic, or does not match that extension, the extension wins. An unknown
    extension, or a declared type that already matches (parameters such as
    ``; charset=utf-8`` are ignored for the comparison), leaves the declared
    value untouched, so correctly labeled attachments are sent exactly as before.
    """
    declared = (mime_type or "").strip()
    accepted = _EXTENSION_MIME_TYPES.get(_file_extension(filename))
    if not accepted:
        return declared
    base_type = declared.split(";", 1)[0].strip().lower()
    if base_type in accepted:
        return declared
    return accepted[0]


def is_pdf(mime_type: str) -> bool:
    """Check if the MIME type represents a PDF document."""
    return mime_type.lower() == "application/pdf"


def is_image(mime_type: str) -> bool:
    """Check if the MIME type represents a supported image format."""
    return mime_type.lower() in IMAGE_MIME_TYPES


def is_tiff(mime_type: str) -> bool:
    """Check if the MIME type represents a TIFF image."""
    return mime_type.lower() in TIFF_MIME_TYPES


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


@asynccontextmanager
async def _stream_download(url: str, *, max_size: int = 0):
    """Stream an HTTP download with size enforcement.

    Yields the validated response object. Checks Content-Length upfront
    and raises ValueError if the file is known to exceed the limit.

    Args:
        url: The URL to download from.
        max_size: Maximum allowed file size in bytes. 0 means unlimited.

    Yields:
        The httpx response object, ready for streaming via aiter_bytes().

    Raises:
        ValueError: If the file exceeds max_size (Content-Length check).
        httpx.HTTPStatusError: If the HTTP request fails.
    """
    async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

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

            yield response


async def stream_tiff_to_content_blocks(
    url: str, *, max_size: int = 0
) -> list[DataContentBlock]:
    """Download a TIFF via streaming and convert each page to a content block.

    Streams the HTTP response directly into a buffer for PIL, enforcing
    size limits as chunks arrive. Each TIFF page is converted to PNG,
    base64-encoded, and wrapped in a DataContentBlock immediately so
    the raw PNG bytes can be freed.

    Args:
        url: The URL to download from.
        max_size: Maximum allowed file size in bytes. 0 means unlimited.

    Returns:
        A list of DataContentBlock instances, one per TIFF page.

    Raises:
        ValueError: If the file exceeds max_size.
        httpx.HTTPStatusError: If the HTTP request fails.
    """
    from langchain_core.messages.content import create_image_block
    from PIL import Image, ImageSequence

    async with _stream_download(url, max_size=max_size) as response:
        buf = io.BytesIO()
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if max_size > 0 and total > max_size:
                raise ValueError(
                    f"File exceeds the {_format_mb(max_size, decimals=0)}"
                    f" limit for LLM payloads"
                    f" (downloaded {_format_mb(total)} so far)"
                )
            buf.write(chunk)

    buf.seek(0)
    blocks: list[DataContentBlock] = []
    with Image.open(buf) as img:
        for frame in ImageSequence.Iterator(img):
            png_buf = io.BytesIO()
            frame.convert("RGBA").save(png_buf, format="PNG")
            png_b64 = base64.b64encode(png_buf.getvalue()).decode("ascii")
            blocks.append(create_image_block(base64=png_b64, mime_type="image/png"))
    return blocks


async def download_file_base64(url: str, *, max_size: int = 0) -> str:
    """Download a file from a URL and return its content as a base64 string.

    Args:
        url: The URL to download from.
        max_size: Maximum allowed file size in bytes. 0 means unlimited.

    Raises:
        ValueError: If the file exceeds max_size.
        httpx.HTTPStatusError: If the HTTP request fails.
    """
    async with _stream_download(url, max_size=max_size) as response:
        return await encode_streamed_base64(response.aiter_bytes(), max_size=max_size)
