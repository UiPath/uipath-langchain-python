"""Abstract base class for model payload handlers."""

from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage


class ModelPayloadHandler(ABC):
    """Abstract base class for handling model-specific LLM parameters.

    Each handler provides provider-specific parameter values for LLM operations.
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
