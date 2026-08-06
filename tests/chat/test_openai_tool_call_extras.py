from collections.abc import Iterator
from typing import Any

from langchain.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from uipath_langchain.agent.messages.message_utils import replace_tool_calls
from uipath_langchain.chat.openai import (
    _OPENAI_TOOL_CALL_EXTRAS_KEY,
    UiPathChatOpenAI,
)


def _client() -> UiPathChatOpenAI:
    return UiPathChatOpenAI.model_construct(
        model_name="provider-model",
        output_version=None,
        use_responses_api=False,
    )


def _completion(tool_calls: list[dict[str, Any]]) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "response-1",
            "model": "provider-model",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }
    )


def test_provider_specific_tool_call_fields_survive_round_trip() -> None:
    client = _client()
    completion = _completion(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "log",
                    "arguments": '{"message":"original"}',
                },
                "provider_metadata": {"opaque_token": "token-1"},
                "provider_flag": True,
            }
        ]
    )

    message = client._create_chat_result(completion).generations[0].message
    assert isinstance(message, AIMessage)
    assert message.additional_kwargs[_OPENAI_TOOL_CALL_EXTRAS_KEY] == {
        "call-1": {
            "provider_metadata": {"opaque_token": "token-1"},
            "provider_flag": True,
        }
    }

    updated_message = replace_tool_calls(
        message,
        [
            ToolCall(
                id="call-1",
                name="log",
                args={"message": "changed"},
                type="tool_call",
            )
        ],
    )
    payload = client._get_request_payload(
        [
            HumanMessage("Log a message"),
            updated_message,
            ToolMessage("done", tool_call_id="call-1"),
        ]
    )

    outgoing_call = payload["messages"][1]["tool_calls"][0]
    assert outgoing_call["function"]["arguments"] == '{"message": "changed"}'
    assert outgoing_call["provider_metadata"] == {"opaque_token": "token-1"}
    assert outgoing_call["provider_flag"] is True


def test_provider_specific_fields_match_multiple_tool_calls_by_id() -> None:
    client = _client()
    completion = _completion(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "first_tool",
                    "arguments": '{"value":"first"}',
                },
                "extra_content": {"google": {"thought_signature": "signature-1"}},
            },
            {
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "second_tool",
                    "arguments": '{"value":"second"}',
                },
                "extra_content": {"google": {"thought_signature": "signature-2"}},
            },
        ]
    )

    message = client._create_chat_result(completion).generations[0].message
    assert isinstance(message, AIMessage)
    reordered_message = replace_tool_calls(
        message,
        [
            ToolCall(
                id="call-2",
                name="second_tool",
                args={"value": "second"},
                type="tool_call",
            ),
            ToolCall(
                id="call-1",
                name="first_tool",
                args={"value": "first"},
                type="tool_call",
            ),
        ],
    )

    payload = client._get_request_payload(
        [
            HumanMessage("Call both tools"),
            reordered_message,
            ToolMessage("second result", tool_call_id="call-2"),
            ToolMessage("first result", tool_call_id="call-1"),
        ]
    )

    outgoing_calls = payload["messages"][1]["tool_calls"]
    assert outgoing_calls[0]["extra_content"] == {
        "google": {"thought_signature": "signature-2"}
    }
    assert outgoing_calls[1]["extra_content"] == {
        "google": {"thought_signature": "signature-1"}
    }


def test_streamed_provider_specific_fields_survive_chunk_merge() -> None:
    client = _client()
    raw_chunks = [
        {
            "id": "response-1",
            "model": "provider-model",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "log",
                                    "arguments": '{"message":',
                                },
                                "extra_content": {
                                    "google": {"thought_signature": "signature-1"}
                                },
                            }
                        ],
                    },
                }
            ],
        },
        {
            "id": "response-1",
            "model": "provider-model",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"original"}'},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "id": "response-1",
            "model": "provider-model",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {},
                }
            ],
        },
    ]

    def generations() -> Iterator[ChatGenerationChunk]:
        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for raw_chunk in raw_chunks:
            chunk = ChatCompletionChunk.model_validate(raw_chunk).model_dump()
            generation = client._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, None
            )
            if generation is not None:
                default_chunk_class = generation.message.__class__
                yield generation

    message = generate_from_stream(generations()).generations[0].message
    assert message.additional_kwargs[_OPENAI_TOOL_CALL_EXTRAS_KEY] == {
        "call-1": {"extra_content": {"google": {"thought_signature": "signature-1"}}}
    }

    payload = client._get_request_payload(
        [
            HumanMessage("Log a message"),
            message,
            ToolMessage("done", tool_call_id="call-1"),
        ]
    )
    assert payload["messages"][1]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "signature-1"}
    }


def test_stream_final_completion_does_not_duplicate_tool_call_extras() -> None:
    client = _client()
    completion = _completion(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "log", "arguments": "{}"},
                "provider_signature": "signature-1",
            }
        ]
    )

    final_chunk = client._get_generation_chunk_from_completion(completion)

    assert _OPENAI_TOOL_CALL_EXTRAS_KEY not in final_chunk.message.additional_kwargs
