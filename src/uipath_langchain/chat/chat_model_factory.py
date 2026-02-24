from typing import Literal

from uipath_langchain_client.base_client import UiPathBaseChatModel
from uipath_langchain_client.factory import get_chat_model as get_chat_model_factory


def get_chat_model(
    model: str,
    *,
    client_type: Literal["passthrough", "normalized"] = "passthrough",
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> UiPathBaseChatModel:
    client_kwargs = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    llm = get_chat_model_factory(model, client_type=client_type, **client_kwargs)
    return llm
