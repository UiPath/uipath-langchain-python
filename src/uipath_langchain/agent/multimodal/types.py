"""Types and constants for multimodal LLM input handling."""

from dataclasses import dataclass


@dataclass
class FileInfo:
    """File information for LLM file attachments."""

    url: str
    name: str
    mime_type: str


IMAGE_MIME_TYPES: set[str] = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}

TEXT_MIME_TYPES: set[str] = {
    "text/plain",
    "text/csv",
    "text/xml",
    "text/html",
    "text/markdown",
    "text/tab-separated-values",
    "application/xml",
    "application/json",
    "application/x-yaml",
}
