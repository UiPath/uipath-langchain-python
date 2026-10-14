"""Tests for the attachment ids trace baggage entry."""

import base64
import json
from unittest.mock import patch

import httpx

from uipath_langchain.chat._legacy.openai import _inject_trace_context_headers
from uipath_langchain.chat.attachment_baggage import (
    llm_call_attachment_ids,
    trace_baggage,
)
from uipath_langchain.chat.chat_model_factory import _TraceContextHeadersCallback


def _decode(entry: str) -> dict[str, str]:
    key, value = entry.split("=", 1)
    assert key == "attachmentIds"
    assert "," not in value
    assert "=" not in value
    return json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))


class TestTraceBaggage:
    def test_returns_the_entries_unchanged_outside_an_llm_call_with_files(self) -> None:
        assert trace_baggage("source=agents") == ["source=agents"]

    def test_adds_the_attachment_ids_inside_the_block(self) -> None:
        ids = {"cv-harmful-content-pdf": "8b0b5e0c-6a43-4c2e-9d0f-3f1f8e2d7a10"}

        with llm_call_attachment_ids(ids):
            baggage = trace_baggage("source=agents")

        assert baggage[0] == "source=agents"
        assert _decode(baggage[1]) == ids

    def test_adds_nothing_for_an_empty_map(self) -> None:
        with llm_call_attachment_ids({}):
            assert trace_baggage("source=agents") == ["source=agents"]

    def test_restores_the_outer_value_after_the_block(self) -> None:
        with llm_call_attachment_ids({"a-txt": "id-a"}):
            with llm_call_attachment_ids({"b-txt": "id-b"}):
                assert _decode(trace_baggage()[0]) == {"b-txt": "id-b"}
            assert _decode(trace_baggage()[0]) == {"a-txt": "id-a"}
        assert trace_baggage() == []


class TestHeaderInjectionCarriesTheAttachmentIds:
    def test_new_client_callback(self) -> None:
        with (
            patch(
                "uipath_langchain.chat.chat_model_factory.build_trace_context_headers",
                return_value={},
            ) as build,
            llm_call_attachment_ids({"report-pdf": "id-1"}),
        ):
            _TraceContextHeadersCallback().on_chat_model_start(
                serialized={}, messages=[[]]
            )

        extra = build.call_args.kwargs["extra_baggage"]
        assert extra[0] == "source=agents"
        assert _decode(extra[1]) == {"report-pdf": "id-1"}

    def test_legacy_client_transport(self) -> None:
        request = httpx.Request("POST", "https://example.com")
        with (
            patch(
                "uipath_langchain.chat._legacy.openai.build_trace_context_headers",
                return_value={},
            ) as build,
            llm_call_attachment_ids({"report-pdf": "id-1"}),
        ):
            _inject_trace_context_headers(request)

        extra = build.call_args.kwargs["extra_baggage"]
        assert _decode(extra[1]) == {"report-pdf": "id-1"}
