"""Compatibility exports for job attachment utilities."""

from ..job_attachments import (
    _attachment_id_uuid_error,
    _create_job_attachment_error_message,
    _describe_validation_errors,
    get_job_attachment_paths,
    get_job_attachments,
    parse_attachments_from_conversation_messages,
    raise_for_job_attachment_error,
    replace_job_attachment_ids,
)

__all__ = [
    "_attachment_id_uuid_error",
    "_create_job_attachment_error_message",
    "_describe_validation_errors",
    "get_job_attachment_paths",
    "get_job_attachments",
    "parse_attachments_from_conversation_messages",
    "raise_for_job_attachment_error",
    "replace_job_attachment_ids",
]
