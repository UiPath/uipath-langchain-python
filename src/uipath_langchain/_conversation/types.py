"""Shared conversational contracts."""

from typing import Any, TypedDict


class ClientSideToolInfo(TypedDict):
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None
