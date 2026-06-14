"""Guards for the mcp SDK upgrade (PR-A).

The MCP client negotiates ``LATEST_PROTOCOL_VERSION`` on initialize, so pinning the SDK to a
version that ships ``2025-11-25`` is what lets UiPath MCP servers offer tasks. These tests fail
loudly if the dependency is ever downgraded below a tasks-capable release.
"""

from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    CreateTaskResult,
    GetTaskResult,
    Result,
    TaskMetadata,
)


def test_sdk_negotiates_2025_11_25() -> None:
    # The client requests LATEST_PROTOCOL_VERSION; tasks require 2025-11-25.
    assert LATEST_PROTOCOL_VERSION == "2025-11-25"


def test_sdk_exposes_task_types() -> None:
    # Task result types must be importable — the suspend-on-UiPath-task feature reads them.
    assert CreateTaskResult is not None
    assert GetTaskResult is not None
    assert TaskMetadata is not None


def test_result_exposes_meta() -> None:
    # The UiPath-job marker rides the result's _meta; the client reads it to detect a UiPath job.
    assert "meta" in Result.model_fields
