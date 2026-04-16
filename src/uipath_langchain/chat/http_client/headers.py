"""Shared UiPath HTTP headers for LLM Gateway requests."""

import os
from urllib.parse import quote

from uipath.platform.common._config import UiPathConfig
from uipath.platform.common.constants import (
    ENV_FOLDER_KEY,
    ENV_JOB_KEY,
    ENV_ORGANIZATION_ID,
    ENV_PROCESS_KEY,
    ENV_TENANT_ID,
    ENV_UIPATH_TRACE_ID,
    HEADER_AGENTHUB_CONFIG,
    HEADER_FOLDER_KEY,
    HEADER_INTERNAL_ACCOUNT_ID,
    HEADER_INTERNAL_TENANT_ID,
    HEADER_JOB_KEY,
    HEADER_LICENSING_CONTEXT,
    HEADER_LLMGATEWAY_BYO_CONNECTION_ID,
    HEADER_MODE,
    HEADER_PROCESS_KEY,
    HEADER_TRACE_ID,
)


def build_uipath_headers(
    *,
    agenthub_config: str | None = None,
    byo_connection_id: str | None = None,
    mode: str | None = None,
    inject_routing: bool = False,
) -> dict[str, str]:
    """Build common UiPath headers for LLM Gateway requests.

    Reads process_key, job_key, folder_key, and trace_id directly from
    environment variables when set.

    Args:
        agenthub_config: Optional AgentHub configuration identifier.
        byo_connection_id: Optional BYO connection identifier.
        mode: Optional agent mode identifier forwarded to AgentHub via
            the X-UiPath-Mode header. Expected values such as
            "standard" or "advanced" are surfaced to Licensing as a raw data
            parameter so advanced-mode runs can be billed differently.
        inject_routing: When True, adds tenant and account routing
            headers that are normally injected by the platform routing
            layer.  Set this when using a service URL override that
            bypasses the platform.
    """
    headers: dict[str, str] = {}
    if agenthub_config:
        headers[HEADER_AGENTHUB_CONFIG] = agenthub_config
    if byo_connection_id:
        headers[HEADER_LLMGATEWAY_BYO_CONNECTION_ID] = byo_connection_id
    if mode:
        headers[HEADER_MODE] = mode
    if process_key := os.getenv(ENV_PROCESS_KEY):
        headers[HEADER_PROCESS_KEY] = quote(process_key, safe="")
    if job_key := os.getenv(ENV_JOB_KEY):
        headers[HEADER_JOB_KEY] = job_key
    if folder_key := os.getenv(ENV_FOLDER_KEY):
        headers[HEADER_FOLDER_KEY] = folder_key
    if trace_id := os.getenv(ENV_UIPATH_TRACE_ID):
        headers[HEADER_TRACE_ID] = trace_id
    if licensing_context := UiPathConfig.licensing_context:
        headers[HEADER_LICENSING_CONTEXT] = licensing_context

    if inject_routing:
        if tenant_id := os.getenv(ENV_TENANT_ID):
            headers[HEADER_INTERNAL_TENANT_ID] = tenant_id
        if organization_id := os.getenv(ENV_ORGANIZATION_ID):
            headers[HEADER_INTERNAL_ACCOUNT_ID] = organization_id

    return headers
