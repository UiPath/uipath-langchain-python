"""Shared client-side tool types."""

from typing import Any, TypedDict


class ClientSideToolInfo(TypedDict):
    """Schemas exposed for a client-side tool."""

    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None


__all__ = ["ClientSideToolInfo"]
