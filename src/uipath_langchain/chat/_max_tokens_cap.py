from collections.abc import Mapping
from typing import Any

from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentStartupError,
    AgentStartupErrorCode,
)


def raise_if_max_tokens_over_cap(
    model: str,
    max_tokens: int | None,
    model_details: Mapping[str, Any] | None,
) -> None:
    cap = (model_details or {}).get("maxOutputTokens")
    if not max_tokens or not isinstance(cap, int) or cap <= 0:
        return
    if max_tokens > cap:
        raise AgentStartupError(
            code=AgentStartupErrorCode.LLM_MAX_TOKENS_EXCEEDED,
            title="Max tokens is higher than the model allows",
            detail=(
                f"The agent's max tokens setting is {max_tokens}, but model "
                f"'{model}' returns at most {cap} output tokens. Set max tokens "
                f"to {cap} or lower in the agent settings."
            ),
            category=UiPathErrorCategory.USER,
        )
