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

# LLM providers enforce payload limits (OpenAI ~20MB, Anthropic ~32MB).
# Base64 encoding adds ~33% overhead, so 15MB raw ≈ 20MB encoded.
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024
