"""Types and constants for multimodal LLM input handling."""

from dataclasses import dataclass

MAX_FILE_SIZE_BYTES: int = 30 * 1024 * 1024  # 30MB

IMAGE_MIME_TYPES: set[str] = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}


@dataclass
class FileInfo:
    """File information for LLM file attachments."""

    url: str
    name: str
    mime_type: str
