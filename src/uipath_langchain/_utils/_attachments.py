"""Shared rendering of the attachment block handed to the model."""

import json
from typing import Any

ATTACHMENTS_BLOCK_PREFIX = "<uip:attachments>"
ATTACHMENTS_BLOCK_SUFFIX = "</uip:attachments>"

# the model copies these straight into tool arguments, which are validated
# against JOB_ATTACHMENT_DEFINITION
_JOB_ATTACHMENT_KEYS = {
    "id": "ID",
    "full_name": "FullName",
    "mime_type": "MimeType",
    "file_path": "FilePath",
}


def render_attachments_block(attachments: list[dict[str, Any]]) -> str:
    """Render attachment references as the text block the model reads."""
    renamed = [
        {_JOB_ATTACHMENT_KEYS.get(key, key): value for key, value in attachment.items()}
        for attachment in attachments
    ]
    return f"{ATTACHMENTS_BLOCK_PREFIX}{json.dumps(renamed)}{ATTACHMENTS_BLOCK_SUFFIX}"
