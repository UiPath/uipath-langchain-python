"""Message content builder for AWS Bedrock Converse API."""

from typing import Any

from .base import (
    MessageContentBuilder,
    download_file_base64,
    download_file_bytes,
    is_image,
    is_pdf,
    sanitize_filename_for_anthropic,
)


class BedrockConverseBuilder(MessageContentBuilder):
    """Builder for AWS Bedrock Converse API content parts.

    PDFs use raw bytes, images use base64-encoded content.
    """

    async def build_file_content_part(
        self,
        url: str,
        filename: str,
        mime_type: str,
    ) -> dict[str, Any]:
        """Build content part for Bedrock Converse API."""
        if is_pdf(mime_type):
            file_bytes = await download_file_bytes(url)
            name = filename.rsplit(".", 1)[0] if "." in filename else filename
            sanitized_name = sanitize_filename_for_anthropic(name)
            return {
                "type": "document",
                "document": {
                    "format": "pdf",
                    "name": sanitized_name,
                    "source": {
                        "bytes": file_bytes,
                    },
                    "citations": {"enabled": True},
                },
            }

        if is_image(mime_type):
            base64_content = await download_file_base64(url)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_content,
                },
            }

        raise ValueError(f"Unsupported mime_type: {mime_type}")
