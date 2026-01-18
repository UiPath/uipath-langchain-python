"""Message content builder for Google Vertex AI Gemini API."""

from typing import Any

from .base import (
    MessageContentBuilder,
    download_file_base64,
    is_image,
    is_pdf,
)


class VertexGeminiBuilder(MessageContentBuilder):
    """Builder for Google Vertex AI Gemini API content parts."""

    async def build_file_content_part(
        self,
        url: str,
        filename: str,
        mime_type: str,
    ) -> dict[str, Any]:
        """Build content part for Vertex Gemini API with downloaded file."""
        if is_image(mime_type) or is_pdf(mime_type):
            base64_content = await download_file_base64(url)
            return {
                "type": "file",
                "source_type": "base64",
                "mime_type": mime_type,
                "data": base64_content,
            }

        raise ValueError(f"Unsupported mime_type: {mime_type}")
