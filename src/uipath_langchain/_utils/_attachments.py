"""Shared rendering of the attachment block handed to the model."""

import json
from typing import Any

ATTACHMENTS_BLOCK_PREFIX = "<uip:attachments>"
ATTACHMENTS_BLOCK_SUFFIX = "</uip:attachments>"


def render_attachments_block(attachments: list[dict[str, Any]]) -> str:
    """Render attachment references as the text block the model reads."""
    return (
        f"{ATTACHMENTS_BLOCK_PREFIX}{json.dumps(attachments)}{ATTACHMENTS_BLOCK_SUFFIX}"
    )
