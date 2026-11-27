"""MIME types for attachments the agent names by file name."""

import mimetypes
from pathlib import Path
from typing import Any

from uipath.platform.attachments import Attachment

from uipath_langchain.agent.multimodal.utils import normalize_mime_type

_DEFAULT_MIME_TYPE = "application/octet-stream"

# mimetypes has no entry for these on every supported Python.
_EXTRA_MIME_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".jsonl": "application/jsonl",
}


def guess_mime_type(file_name: str) -> str:
    """Resolve a file's MIME type from its extension."""
    suffix = Path(file_name).suffix.lower()
    if suffix in _EXTRA_MIME_TYPES:
        return _EXTRA_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(file_name)
    return normalize_mime_type(guessed or _DEFAULT_MIME_TYPE, file_name)


def with_guardrail_mime_type(attachment: Any) -> Any:
    """The attachment with a type a guardrail reference can carry.

    A file uploaded with no type keeps an empty one, and a reference without a
    type is dropped (``uipath.platform.guardrails.to_references``), so a
    files-only guardrail would pass without seeing it. It is given its
    extension's type, else ``application/octet-stream``, which the Agents
    backend resolves from the content. Anything but an ``Attachment`` is
    returned as it is.
    """
    if not isinstance(attachment, Attachment) or attachment.mime_type:
        return attachment
    mime_type = normalize_mime_type("", attachment.full_name) or _DEFAULT_MIME_TYPE
    return attachment.model_copy(update={"mime_type": mime_type})
