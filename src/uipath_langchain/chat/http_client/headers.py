"""Shared UiPath HTTP headers for LLM Gateway requests."""

import os
from urllib.parse import quote


def build_uipath_headers(
    token: str,
    *,
    agenthub_config: str | None = None,
    byo_connection_id: str | None = None,
) -> dict[str, str]:
    """Build common UiPath headers for LLM Gateway requests.

    Reads process_key, job_key, folder_key, and trace_id directly from
    environment variables when set.
    """
    headers: dict[str, str] = {
        "Authorization": f"Bearer {token}",
    }
    if agenthub_config:
        headers["X-UiPath-AgentHub-Config"] = agenthub_config
    if byo_connection_id:
        headers["X-UiPath-LlmGateway-ByoIsConnectionId"] = byo_connection_id
    if process_key := os.getenv("UIPATH_PROCESS_KEY"):
        headers["X-UiPath-ProcessKey"] = quote(process_key, safe="")
    if job_key := os.getenv("UIPATH_JOB_KEY"):
        headers["x-uipath-jobkey"] = job_key
    if folder_key := os.getenv("UIPATH_FOLDER_KEY"):
        headers["x-uipath-folderkey"] = folder_key
    if trace_id := os.getenv("UIPATH_TRACE_ID"):
        headers["x-uipath-traceid"] = trace_id
    return headers
