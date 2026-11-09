"""Abstract base class for model payload handlers."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool


def is_byo_model(model: Any) -> bool:
    settings = getattr(model, "client_settings", None)
    if settings is None:
        return False
    try:
        info = settings.get_model_info(
            model.model_name,
            byo_connection_id=getattr(model, "byo_connection_id", None),
        )
    except Exception:
        return True
    return info.get("modelSubscriptionType") != "UiPathOwned"


class ModelPayloadHandler(ABC):
    """Abstract base class for handling model-specific LLM parameters.

    Each handler provides provider-specific parameter values for LLM operations.
    """

    def __init__(self, model: BaseChatModel):
        self.model = model

    @abstractmethod
    def get_tool_binding_kwargs(
        self,
        tools: Sequence[BaseTool],
        tool_choice: Literal["auto", "any"],
        parallel_tool_calls: bool | None = None,
        strict_mode: bool | None = None,
    ) -> dict[str, Any]: ...

    def apply_prompt_caching(
        self, messages: list[AnyMessage]
    ) -> tuple[list[AnyMessage], dict[str, Any]]:
        return messages, {}

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
        parallel_tool_calls: bool | None = None,
        strict_mode: bool | None = None,
    ) -> dict[str, Any]:
        return {"tool_choice": tool_choice}

    def check_stop_reason(self, response: AIMessage) -> None:
        return
