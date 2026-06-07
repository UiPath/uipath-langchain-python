"""Payload handler for Fireworks AI."""

from typing import Any, Literal, Sequence

from langchain_core.tools import BaseTool

from .base import ModelPayloadHandler


class FireworksPayloadHandler(ModelPayloadHandler):
    """Payload handler for Fireworks AI API."""

    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool | None = None,
        strict_mode: bool | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"tool_choice": tool_choice}
        if strict_mode is True:
            kwargs["strict"] = True
        return kwargs

    def check_stop_reason(self, response: Any) -> None:
        return
