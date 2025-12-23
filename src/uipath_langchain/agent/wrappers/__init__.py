"""Wrappers to add behavior to tools while keeping them graph agnostic."""

from .job_attachment_wrapper import get_job_attachment_wrapper
from .static_args_wrapper import get_static_args_wrapper

__all__ = ["get_static_args_wrapper", "get_job_attachment_wrapper"]
