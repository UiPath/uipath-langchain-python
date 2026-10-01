"""MIME types for attachments the agent names by file name."""

import mimetypes
from pathlib import Path

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
