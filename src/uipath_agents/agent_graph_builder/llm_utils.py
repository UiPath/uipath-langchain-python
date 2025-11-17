from uipath_langchain.chat.models import UiPathChat

LLM_TIMEOUT_SECONDS = 300
LLM_MAX_RETRIES = 2


def create_llm(
    model: str,
    temperature: float = 0,
    max_tokens: int | None = 16_384,
    timeout: int = LLM_TIMEOUT_SECONDS,
    max_retries: int = LLM_MAX_RETRIES,
    disable_streaming: bool = True,
) -> UiPathChat:
    """Create and configure UiPathChat LLM instance."""

    llm = UiPathChat(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=timeout,
        max_retries=max_retries,
        disable_streaming=disable_streaming,
    )

    return llm
