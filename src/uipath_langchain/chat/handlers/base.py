"""Abstract base class for model payload handlers."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage


class ModelPayloadHandler(ABC):
    """Abstract base class for handling model-specific LLM parameters.

    Each handler provides provider-specific parameter values for LLM operations.
    """

    @abstractmethod
    def get_required_tool_choice(self) -> str | dict[str, Any]:
        """Get the tool_choice value that enforces tool usage.

        Returns:
            Provider-specific value to force tool usage:
            - "required" for OpenAI-compatible models
            - "any" for Bedrock Converse and Vertex models (string format)
            - {"type": "any"} for Bedrock Invoke API (dict format required)
        """

    @abstractmethod
    def check_stop_reason(self, response: AIMessage) -> None:
        """Check response stop reason and raise exception for faulty terminations.

        Override in subclasses to implement provider-specific stop reason validation.

        Args:
            response: The AIMessage response from the model

        Raises:
            AgentTerminationException: If stop reason indicates a faulty termination
        """
