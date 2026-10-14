"""Trace baggage naming the Orchestrator attachment behind each file inlined into an LLM call.

The LLM Gateway forwards the caller's trace baggage unchanged to the guardrails service, which
uses this entry to link the inline files it evaluated back to their Orchestrator attachments, so
centralized guardrail spans can offer them for download.
"""

import base64
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

ATTACHMENT_IDS_BAGGAGE_KEY = "attachmentIds"

_attachment_ids: ContextVar[str | None] = ContextVar(
    "uipath_llm_call_attachment_ids", default=None
)


@contextmanager
def llm_call_attachment_ids(ids_by_filename: Mapping[str, str]) -> Iterator[None]:
    """Name, for the LLM calls made inside the block, the attachment behind each inline file.

    Args:
        ids_by_filename: The ``filename`` each inline file part carries, mapped to the id of
            the Orchestrator attachment the part was read from.
    """
    token = _attachment_ids.set(_encode(ids_by_filename) if ids_by_filename else None)
    try:
        yield
    finally:
        _attachment_ids.reset(token)


def trace_baggage(*entries: str) -> list[str]:
    """Return ``entries`` plus the attachment ids entry when the current LLM call inlines files."""
    encoded = _attachment_ids.get()
    if not encoded:
        return list(entries)
    return [*entries, f"{ATTACHMENT_IDS_BAGGAGE_KEY}={encoded}"]


def _encode(ids_by_filename: Mapping[str, str]) -> str:
    # Unpadded base64url keeps the baggage delimiters (',' and '=') out of the value.
    payload = json.dumps(dict(ids_by_filename), separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
