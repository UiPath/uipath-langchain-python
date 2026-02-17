"""Abstract base class for model payload handlers."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage


class ModelPayloadHandler(ABC):
    """Abstract base class for handling model-specific LLM parameters.

    Each handler provides provider-specific parameter values for LLM operations.
    """

    def get_parallel_tool_calls_kwargs(
        self, parallel_tool_calls: bool
    ) -> dict[str, Any]:
        """Get provider-specific bind_tools kwargs for controlling parallel tool calls.

        Returns:
            Dict of kwargs to spread into model.bind_tools().
            Empty dict if the provider doesn't support this parameter.
        """
        return {}

    @abstractmethod
    def check_stop_reason(self, response: AIMessage) -> None:
        """Check response stop reason and raise exception for faulty terminations.

        Override in subclasses to implement provider-specific stop reason validation.

        Args:
            response: The AIMessage response from the model

        Raises:
            ChatModelError: If stop reason indicates a faulty termination
        """
