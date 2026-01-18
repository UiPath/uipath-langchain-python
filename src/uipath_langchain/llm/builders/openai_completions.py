"""Message content builder for OpenAI Completions API."""

from typing import Any

from .base import (
    MessageContentBuilder,
    download_file_base64,
    is_image,
    is_pdf,
)


class OpenAICompletionsBuilder(MessageContentBuilder):
    """Builder for OpenAI Completions API content parts.

    Note: PDFs are not supported by the OpenAI Completions API.
    """

    async def build_file_content_part(
        self,
        url: str,
        filename: str,
        mime_type: str,
    ) -> dict[str, Any]:
        """Build content part for OpenAI Completions API with downloaded file."""
        if is_pdf(mime_type):
            raise ValueError(
                "PDFs are not supported when using the OpenAI Completions API"
            )

        if is_image(mime_type):
            base64_content = await download_file_base64(url)
            data_url = f"data:{mime_type};base64,{base64_content}"
            return {
                "type": "image_url",
                "image_url": {"url": data_url},
            }

        raise ValueError(f"Unsupported mime_type: {mime_type}")
