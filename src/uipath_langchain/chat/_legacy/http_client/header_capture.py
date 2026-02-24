from contextvars import ContextVar
from typing import Optional

from langchain_core.outputs import ChatGeneration, ChatResult


class HeaderCapture:
    """Captures HTTP response headers and applies them to LangChain generations."""

    def __init__(self, name: str = "response_headers"):
        """Initialize with a new context var.

        Args:
            name: Name for the context var."""
        self._headers: ContextVar[Optional[dict[str, str]]] = ContextVar(
            name, default=None
        )

    def set(self, headers: dict[str, str]) -> None:
        """Store headers in this instance's context var."""
        self._headers.set(headers)

    def clear(self) -> None:
        """Clear stored headers from this instance's context var."""
        self._headers.set(None)

    def attach_to_chat_generation(
        self, generation: ChatGeneration, metadata_key: str = "headers"
    ) -> None:
        """Attach captured headers to the generation message's response_metadata."""
        headers = self._headers.get()
        if headers:
            generation.message.response_metadata[metadata_key] = headers

    def attach_to_chat_result(
        self, result: ChatResult, metadata_key: str = "headers"
    ) -> ChatResult:
        """Attach captured headers to the message response_metadata of each generation."""
        for generation in result.generations:
            self.attach_to_chat_generation(generation, metadata_key)
        return result
