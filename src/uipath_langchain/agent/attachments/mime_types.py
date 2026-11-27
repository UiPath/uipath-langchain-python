"""MIME types for attachments the agent names by file name."""

import mimetypes
from collections.abc import Mapping
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
    """The attachment, or attachment mention, with a type a guardrail reference can carry.

    A file uploaded with no type keeps an empty one, and a reference without a
    type is dropped (``uipath.platform.guardrails.to_references``), so a
    files-only guardrail would pass without seeing it. An ``Attachment``, or a
    mention mapping that names its file, is given its extension's type, else
    ``application/octet-stream``, which the Agents backend resolves from the
    content. Anything else, and anything that has a type, is returned as it is.
    """
    if isinstance(attachment, Attachment):
        if attachment.mime_type:
            return attachment
        mime_type = _guardrail_fallback_type(attachment.full_name)
        return attachment.model_copy(update={"mime_type": mime_type})
    if not isinstance(attachment, Mapping) or _first_text(
        attachment, ("MimeType", "mime_type")
    ):
        return attachment
    name = _first_text(attachment, ("FullName", "full_name"))
    if not name:
        return attachment
    return {**attachment, "MimeType": _guardrail_fallback_type(name)}


def _guardrail_fallback_type(file_name: str) -> str:
    return normalize_mime_type("", file_name) or _DEFAULT_MIME_TYPE


def _first_text(mapping: Mapping[Any, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return ""
