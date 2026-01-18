"""Message content builder for OpenAI Responses API."""

from typing import Any

from .base import (
    MessageContentBuilder,
    download_file_base64,
    is_image,
    is_pdf,
)


class OpenAIResponsesBuilder(MessageContentBuilder):
    """Builder for OpenAI Responses API content parts."""

    async def build_file_content_part(
        self,
        url: str,
        filename: str,
        mime_type: str,
    ) -> dict[str, Any]:
        """Build content part for OpenAI Responses API with downloaded file."""
        base64_content = await download_file_base64(url)

        if is_image(mime_type):
            data_url = f"data:{mime_type};base64,{base64_content}"
            return {
                "type": "input_image",
                "image_url": data_url,
            }

        if is_pdf(mime_type):
            data = f"data:application/pdf;base64,{base64_content}"
            return {
                "type": "file",
                "file": {
                    "filename": filename,
                    "file_data": data,
                },
            }

        raise ValueError(f"Unsupported mime_type: {mime_type}")
