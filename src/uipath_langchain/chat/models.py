import json
import logging
from typing import Any, AsyncIterator, Iterator, Literal, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_openai.chat_models import AzureChatOpenAI
from pydantic import BaseModel
from uipath.utils import EndpointManager

from uipath_langchain._utils._request_mixin import UiPathRequestMixin

logger = logging.getLogger(__name__)


class UiPathAzureChatOpenAI(UiPathRequestMixin, AzureChatOpenAI):
    """Custom LLM connector for LangChain integration with UiPath."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if "tools" in kwargs and not kwargs["tools"]:
            del kwargs["tools"]

        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        response = self._call(self.url, payload, self.auth_headers)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if "tools" in kwargs and not kwargs["tools"]:
            del kwargs["tools"]

        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        response = await self._acall(self.url, payload, self.auth_headers)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if "tools" in kwargs and not kwargs["tools"]:
            del kwargs["tools"]
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        default_chunk_class = AIMessageChunk

        for chunk in self._stream_request(self.url, payload, self.auth_headers):
            if self.logger:
                self.logger.debug(f"[Stream] Got chunk from _stream_request: {chunk}")
            generation_chunk = self._convert_chunk(
                chunk, default_chunk_class, include_tool_calls=True
            )
            if generation_chunk is None:
                if self.logger:
                    self.logger.debug("[Stream] Skipping None generation_chunk")
                continue

            if self.logger:
                self.logger.debug(
                    f"[Stream] Yielding generation_chunk: {generation_chunk}"
                )

            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                )

            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if "tools" in kwargs and not kwargs["tools"]:
            del kwargs["tools"]
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        default_chunk_class = AIMessageChunk

        async for chunk in self._astream_request(self.url, payload, self.auth_headers):
            generation_chunk = self._convert_chunk(
                chunk, default_chunk_class, include_tool_calls=True
            )
            if generation_chunk is None:
                continue

            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                )

            yield generation_chunk

    def with_structured_output(
        self,
        schema: Any = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Model wrapper that returns outputs formatted to match the given schema."""
        schema = (
            schema.model_json_schema()
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else schema
        )
        return super().with_structured_output(
            schema=schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )

    @property
    def endpoint(self) -> str:
        endpoint = EndpointManager.get_passthrough_endpoint()
        logger.debug("Using endpoint: %s", endpoint)
        return endpoint.format(
            model=self.model_name, api_version=self.openai_api_version
        )


class UiPathChat(UiPathRequestMixin, AzureChatOpenAI):
    """Custom LLM connector for LangChain integration with UiPath Normalized."""

    def _create_chat_result(
        self,
        response: Union[dict[str, Any], BaseModel],
        generation_info: dict[Any, Any] | None = None,
    ) -> ChatResult:
        if not isinstance(response, dict):
            response = response.model_dump()
        message = response["choices"][0]["message"]
        usage = response["usage"]

        ai_message = AIMessage(
            content=message.get("content", ""),
            usage_metadata=UsageMetadata(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            additional_kwargs={},
            response_metadata={
                "token_usage": response["usage"],
                "model_name": self.model_name,
                "finish_reason": response["choices"][0].get("finish_reason", None),
                "system_fingerprint": response["id"],
                "created": response["created"],
            },
        )

        if "tool_calls" in message:
            ai_message.tool_calls = [
                {
                    "id": tool["id"],
                    "name": tool["name"],
                    "args": tool["arguments"],
                    "type": "tool_call",
                }
                for tool in message["tool_calls"]
            ]
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[Any, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # hacks to make the request work with uipath normalized
        for message in payload["messages"]:
            if message["content"] is None:
                message["content"] = ""
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    tool_call["name"] = tool_call["function"]["name"]
                    tool_call["arguments"] = json.loads(
                        tool_call["function"]["arguments"]
                    )
            if message["role"] == "tool":
                message["content"] = {
                    "result": message["content"],
                    "call_id": message["tool_call_id"],
                }
        return payload

    def _normalize_tool_choice(self, kwargs: dict[str, Any]) -> None:
        """Normalize tool_choice for UiPath Gateway compatibility.

        Converts LangChain tool_choice formats to UiPath Gateway format:
        - String "required" -> {"type": "required"}
        - String "auto" -> {"type": "auto"}
        - Dict with function -> {"type": "tool", "name": "function_name"}
        """
        if "tool_choice" in kwargs:
            tool_choice = kwargs["tool_choice"]

            if isinstance(tool_choice, str):
                if tool_choice in ("required", "auto", "none"):
                    logger.debug(
                        f"Converting tool_choice from '{tool_choice}' to {{'type': '{tool_choice}'}}"
                    )
                    kwargs["tool_choice"] = {"type": tool_choice}
            elif (
                isinstance(tool_choice, dict) and tool_choice.get("type") == "function"
            ):
                function_name = tool_choice["function"]["name"]
                logger.debug(
                    f"Converting tool_choice from function '{function_name}' to tool format"
                )
                kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": function_name,
                }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        if kwargs.get("tools"):
            kwargs["tools"] = [tool["function"] for tool in kwargs["tools"]]
        self._normalize_tool_choice(kwargs)

        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        response = self._call(self.url, payload, self.auth_headers)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        if kwargs.get("tools"):
            kwargs["tools"] = [tool["function"] for tool in kwargs["tools"]]
        self._normalize_tool_choice(kwargs)

        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        response = await self._acall(self.url, payload, self.auth_headers)
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the LLM on a given prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
            run_manager: A run manager with callbacks for the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            An iterator of ChatGenerationChunk objects.
        """
        if kwargs.get("tools"):
            kwargs["tools"] = [tool["function"] for tool in kwargs["tools"]]
        self._normalize_tool_choice(kwargs)
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        default_chunk_class = AIMessageChunk

        for chunk in self._stream_request(self.url, payload, self.auth_headers):
            if self.logger:
                self.logger.debug(f"[Stream] Got chunk from _stream_request: {chunk}")
            generation_chunk = self._convert_chunk(
                chunk, default_chunk_class, include_tool_calls=True
            )
            if generation_chunk is None:
                if self.logger:
                    self.logger.debug("[Stream] Skipping None generation_chunk")
                continue

            if self.logger:
                self.logger.debug(
                    f"[Stream] Yielding generation_chunk: {generation_chunk}"
                )

            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                )

            yield generation_chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream the LLM on a given prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
            run_manager: A run manager with callbacks for the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            An async iterator of ChatGenerationChunk objects.
        """
        if kwargs.get("tools"):
            kwargs["tools"] = [tool["function"] for tool in kwargs["tools"]]
        self._normalize_tool_choice(kwargs)
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        # Update headers to enable streaming
        headers = {**self.auth_headers}
        headers["X-UiPath-Streaming-Enabled"] = "true"

        default_chunk_class = AIMessageChunk

        async for chunk in self._astream_request(self.url, payload, headers):
            generation_chunk = self._convert_chunk(
                chunk, default_chunk_class, include_tool_calls=True
            )
            if generation_chunk is None:
                continue

            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                )

            yield generation_chunk

    def with_structured_output(
        self,
        schema: Any = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Any]:
        """Model wrapper that returns outputs formatted to match the given schema."""
        if method == "json_schema" and (
            not self.model_name or not self.model_name.startswith("gpt")
        ):
            method = "function_calling"
            if self.logger:
                self.logger.warning(
                    "The json_schema output is not supported for non-GPT models. Using function_calling instead.",
                    extra={
                        "ActionName": self.settings.action_name,
                        "ActionId": self.settings.action_id,
                    }
                    if self.settings
                    else None,
                )
        schema = (
            schema.model_json_schema()
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else schema
        )
        return super().with_structured_output(
            schema=schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )

    @property
    def endpoint(self) -> str:
        endpoint = EndpointManager.get_normalized_endpoint()
        logger.debug("Using endpoint: %s", endpoint)
        return endpoint.format(
            model=self.model_name, api_version=self.openai_api_version
        )

    @property
    def is_normalized(self) -> bool:
        return True
