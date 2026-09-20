"""The client-side tool schema contract shared by the runtime and the graphs."""

from typing import Any, TypedDict


class ClientSideToolInfo(TypedDict):
    input_schema: dict[str, Any] | None
    output_schema: dict[str, Any] | None
