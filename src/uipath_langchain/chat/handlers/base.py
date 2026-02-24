"""Abstract base class for model payload handlers."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Sequence

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool


class ModelPayloadHandler(ABC):
    """Abstract base class for handling model-specific LLM parameters.

    Each handler provides provider-specific parameter values for LLM operations.
    """

    @abstractmethod
    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool = True,
        strict_mode: bool = False,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def check_stop_reason(self, response: AIMessage) -> None:
        """Check response stop reason and raise exception for faulty terminations.

        Override in subclasses to implement provider-specific stop reason validation.

        Args:
            response: The AIMessage response from the model

        Raises:
            ChatModelError: If stop reason indicates a faulty termination
        """


class DefaultModelPayloadHandler(ModelPayloadHandler):
    """Default model payload handler."""

    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool = True,
        strict_mode: bool = False,
    ) -> dict[str, Any]:
        return {"tool_choice": tool_choice}

    def check_stop_reason(self, response: AIMessage) -> None:
        return
