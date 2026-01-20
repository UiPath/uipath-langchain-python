"""Wrappers to add behavior to tools while keeping them graph agnostic."""

from .job_attachment_wrapper import get_job_attachment_wrapper

__all__ = ["get_job_attachment_wrapper"]
