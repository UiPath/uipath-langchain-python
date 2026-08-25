"""Backward-compatible imports for agent attachment utilities."""

from ..attachments.job_attachments import (
    get_job_attachment_paths,
    get_job_attachments,
    parse_attachments_from_conversation_messages,
    raise_for_job_attachment_error,
    replace_job_attachment_ids,
)

__all__ = [
    "get_job_attachment_paths",
    "get_job_attachments",
    "parse_attachments_from_conversation_messages",
    "raise_for_job_attachment_error",
    "replace_job_attachment_ids",
]
