import base64
from enum import StrEnum
from typing import Any

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

IMAGE_MIME_TYPES: set[str] = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
}


class LlmProvider(StrEnum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    UNKNOWN = "unknown"


def is_pdf(mime_type: str) -> bool:
    """Check if the MIME type represents a PDF document."""
    return mime_type.lower() == "application/pdf"


def is_image(mime_type: str) -> bool:
    """Check if the MIME type represents a supported image format (PNG, JPEG, GIF, WebP)."""
    return mime_type.lower() in IMAGE_MIME_TYPES


def detect_provider(model_name: str) -> LlmProvider:
    """Detect the LLM provider (Bedrock, OpenAI, or Vertex) based on the model name."""
    if not model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    model_lower = model_name.lower()

    if "anthropic" in model_lower or "claude" in model_lower:
        return LlmProvider.BEDROCK

    if "gpt" in model_lower:
        return LlmProvider.OPENAI

    if "gemini" in model_lower:
        return LlmProvider.VERTEX

    raise ValueError(f"Unsupported model: {model_name}")


async def _download_file(url: str) -> str:
    """Download a file from a URL and return its content as a base64 string."""
    async with httpx.AsyncClient(**get_httpx_client_kwargs()) as client:
        response = await client.get(url)
        response.raise_for_status()
        file_content = response.content

    return base64.b64encode(file_content).decode("utf-8")


async def build_message_content_part_from_data(
    url: str,
    filename: str,
    mime_type: str,
    model: str,
) -> dict[str, Any]:
    """Download a file and build a provider-specific message content part.

    The format varies based on the detected provider (Bedrock, OpenAI, or Vertex).
    """
    provider = detect_provider(model)

    if provider == LlmProvider.BEDROCK:
        raise ValueError("Anthropic models are not yet supported for file attachments")

    if provider == LlmProvider.OPENAI:
        return await _build_openai_content_part_from_data(
            url, mime_type, filename, False
        )

    if provider == LlmProvider.VERTEX:
        raise ValueError("Gemini models are not yet supported for file attachments")

    raise ValueError(f"Unsupported provider: {provider}")


async def _build_openai_content_part_from_data(
    url: str,
    mime_type: str,
    filename: str,
    download_image: bool,
) -> dict[str, Any]:
    """Build a content part for OpenAI models (base64-encoded or URL reference)."""
    if download_image:
        base64_content = await _download_file(url)
        if is_image(mime_type):
            data_url = f"data:{mime_type};base64,{base64_content}"
            return {
                "type": "input_image",
                "image_url": data_url,
            }

        if is_pdf(mime_type):
            return {
                "type": "input_file",
                "filename": filename,
                "file_data": base64_content,
            }

    elif is_image(mime_type):
        return {
            "type": "input_image",
            "image_url": url,
        }

    elif is_pdf(mime_type):
        return {
            "type": "input_file",
            "file_url": url,
        }

    raise ValueError(f"Unsupported mime_type: {mime_type}")
