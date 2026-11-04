"""How attachments are shown to the model: the text block and the block tags."""

import json
from typing import Any

ATTACHMENTS_BLOCK_PREFIX = "<uip:attachments>"
ATTACHMENTS_BLOCK_SUFFIX = "</uip:attachments>"

# ``extras`` keys naming the attachment an SDK-built file or image block came from
ATTACHMENT_ID_EXTRA = "uipath_attachment_id"
ATTACHMENT_NAME_EXTRA = "uipath_file_name"

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
    # an attachment name is caller-controlled and would otherwise be able to
    # close this block early. In JSON output "<" only occurs inside a string
    payload = json.dumps(renamed).replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{ATTACHMENTS_BLOCK_PREFIX}{payload}{ATTACHMENTS_BLOCK_SUFFIX}"
