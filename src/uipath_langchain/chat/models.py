import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from uipath_langchain_client.clients.normalized.chat_models import (
    UiPathChat as _UpstreamUiPathChat,
)

from .openai import UiPathAzureChatOpenAI, UiPathChatOpenAI

DEFAULT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"


def _default_factory() -> str:
    return os.getenv("UIPATH_MODEL_NAME", DEFAULT_MODEL_NAME)


def _strip_chunk_metadata(chunk: ChatGenerationChunk, final_seen: bool) -> bool:
    """Strip metadata fields that accumulate via string concatenation in AIMessageChunk.__add__.

    Returns updated final_seen flag.
    """
    gi = chunk.generation_info
    if not gi:
        return final_seen

    has_finish = bool(gi.get("finish_reason"))
    if has_finish:
        if final_seen:
            # Duplicate final chunk (API sometimes sends finish_reason twice): strip all
            for key in ("model_name", "id", "created", "finish_reason"):
                gi.pop(key, None)
        else:
            final_seen = True
    else:
        # Intermediate chunk: strip metadata that string-concatenates across chunks
        for key in ("model_name", "id", "created"):
            gi.pop(key, None)
    return final_seen


class UiPathChat(_UpstreamUiPathChat):
    def _uipath_stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Generator[ChatGenerationChunk, None, None]:
        final_seen = False
        for chunk in super()._uipath_stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            final_seen = _strip_chunk_metadata(chunk, final_seen)
            yield chunk

    async def _uipath_astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatGenerationChunk, None]:
        final_seen = False
        async for chunk in super()._uipath_astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            final_seen = _strip_chunk_metadata(chunk, final_seen)
            yield chunk


UiPathChat.model_fields["model_name"].default_factory = _default_factory
UiPathChat.model_rebuild(force=True)


__all__ = [
    "UiPathChat",
    "UiPathAzureChatOpenAI",
    "UiPathChatOpenAI",
]
