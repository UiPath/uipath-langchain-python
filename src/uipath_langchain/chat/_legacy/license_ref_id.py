"""Process-wide storage for the active LLM-call license ref ID.

The value is set by uipath-agents when a model_run span starts, and read by
each vendor's transport-level injector to attach the
``X-UiPath-License-RefId`` header to outgoing LLM gateway requests. It lives
here rather than next to any one vendor because the same global is shared
across all legacy chat clients (OpenAI, Bedrock, ...).

A plain module global is used because the LangChain callback (which sets the
value) and the transport (which reads it) run on different threads, so
neither ContextVar nor threading.local work.
"""

_current_license_ref_id: str | None = None


def set_license_ref_id(value: str | None) -> None:
    """Set the license ref ID for injection on LLM requests."""
    global _current_license_ref_id
    _current_license_ref_id = value


def get_license_ref_id() -> str | None:
    """Read the current license ref ID."""
    return _current_license_ref_id
