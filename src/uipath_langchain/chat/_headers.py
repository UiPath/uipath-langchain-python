"""Shared UiPath runtime context headers for chat model requests."""

from uipath.platform.common import UiPathConfig
from uipath.platform.common.constants import (
    HEADER_FOLDER_KEY,
    HEADER_JOB_KEY,
    HEADER_TRACE_ID,
)


def build_uipath_context_headers() -> dict[str, str]:
    """Build UiPath runtime context headers for agenthub completions.

    Returns headers for job_key, folder_key, and trace_id when set
    in the environment via UiPathConfig.
    """
    headers: dict[str, str] = {}
    if UiPathConfig.job_key:
        headers[HEADER_JOB_KEY] = UiPathConfig.job_key
    if UiPathConfig.folder_key:
        headers[HEADER_FOLDER_KEY] = UiPathConfig.folder_key
    if UiPathConfig.trace_id:
        headers[HEADER_TRACE_ID] = UiPathConfig.trace_id
    return headers
