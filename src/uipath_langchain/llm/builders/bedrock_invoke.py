"""Message content builder for AWS Bedrock Invoke API."""

from typing import Any

from .base import (
    MessageContentBuilder,
    download_file_base64,
    is_image,
    is_pdf,
)


class BedrockInvokeBuilder(MessageContentBuilder):
    """Builder for AWS Bedrock Invoke API content parts.

    Uses base64-encoded content for both PDFs and images.
    """

    async def build_file_content_part(
        self,
        url: str,
        filename: str,
        mime_type: str,
    ) -> dict[str, Any]:
        """Build content part for Bedrock Invoke API."""
        base64_content = await download_file_base64(url)

        if is_pdf(mime_type):
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_content,
                },
            }

        if is_image(mime_type):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_content,
                },
            }

        raise ValueError(f"Unsupported mime_type: {mime_type}")
