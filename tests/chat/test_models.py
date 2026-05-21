"""Tests for UiPathChat streaming metadata stripping in chat/models.py."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.outputs import ChatGenerationChunk
from uipath_langchain_client.clients.normalized.chat_models import (
    UiPathChat as _UpstreamUiPathChat,
)

from uipath_langchain.chat.models import UiPathChat, _strip_chunk_metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(gi: dict[str, Any] | None) -> MagicMock:
    chunk = MagicMock(spec=ChatGenerationChunk)
    chunk.generation_info = gi
    return chunk


# ---------------------------------------------------------------------------
# TestStripChunkMetadata
# ---------------------------------------------------------------------------


class TestStripChunkMetadata:
    def test_no_generation_info_returns_flag_unchanged_false(self) -> None:
        chunk = _make_chunk(None)
        assert _strip_chunk_metadata(chunk, False) is False

    def test_no_generation_info_returns_flag_unchanged_true(self) -> None:
        chunk = _make_chunk(None)
        assert _strip_chunk_metadata(chunk, True) is True

    def test_empty_gi_returns_flag_unchanged(self) -> None:
        chunk = _make_chunk({})
        assert _strip_chunk_metadata(chunk, False) is False

    def test_intermediate_chunk_strips_metadata_preserves_other_keys(self) -> None:
        gi: dict[str, Any] = {
            "model_name": "gpt-4o",
            "id": "chatcmpl-abc",
            "created": 1234567890,
            "other": "keep_me",
        }
        chunk = _make_chunk(gi)
        result = _strip_chunk_metadata(chunk, False)

        assert result is False
        assert "model_name" not in gi
        assert "id" not in gi
        assert "created" not in gi
        assert gi["other"] == "keep_me"

    def test_first_final_chunk_sets_flag_and_leaves_content_untouched(self) -> None:
        gi: dict[str, Any] = {
            "finish_reason": "stop",
            "model_name": "gpt-4o",
            "id": "chatcmpl-abc",
            "created": 1234567890,
        }
        chunk = _make_chunk(gi)
        result = _strip_chunk_metadata(chunk, False)

        assert result is True
        # First final chunk: only the flag is set, keys are NOT stripped
        assert gi["finish_reason"] == "stop"
        assert gi["model_name"] == "gpt-4o"

    def test_duplicate_final_chunk_strips_all_four_keys(self) -> None:
        gi: dict[str, Any] = {
            "finish_reason": "stop",
            "model_name": "gpt-4o",
            "id": "chatcmpl-abc",
            "created": 1234567890,
        }
        chunk = _make_chunk(gi)
        result = _strip_chunk_metadata(chunk, True)  # final_seen already True

        assert result is True
        assert "finish_reason" not in gi
        assert "model_name" not in gi
        assert "id" not in gi
        assert "created" not in gi

    def test_intermediate_chunk_missing_keys_no_error(self) -> None:
        gi: dict[str, Any] = {"delta": "hello"}
        chunk = _make_chunk(gi)
        result = _strip_chunk_metadata(chunk, False)

        assert result is False
        assert gi == {"delta": "hello"}


# ---------------------------------------------------------------------------
# TestUiPathChatStream
# ---------------------------------------------------------------------------


class TestUiPathChatStream:
    def test_uipath_stream_yields_all_chunks_with_metadata_stripping(self) -> None:
        chunks = [
            _make_chunk(
                {"model_name": "gpt-4o", "id": "1", "created": 0}
            ),  # intermediate
            _make_chunk({"finish_reason": "stop"}),  # final
        ]

        def fake_stream(
            self_inner: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kw: Any,
        ) -> Any:
            yield from chunks

        with patch.dict(
            "os.environ",
            {
                "UIPATH_ACCESS_TOKEN": "eyJhbGciOiAiUlMyNTYiLCAidHlwIjogIkpXVCJ9.eyJzdWIiOiAidGVzdCIsICJleHAiOiA5OTk5OTk5OTk5fQ.fakesignature",
                "UIPATH_URL": "https://dummy.uipath.com",
                "UIPATH_TENANT_ID": "dummy-tenant",
                "UIPATH_ORGANIZATION_ID": "dummy-org",
            },
        ):
            chat = UiPathChat.model_construct()
        with patch.object(_UpstreamUiPathChat, "_uipath_stream", fake_stream):
            result = list(chat._uipath_stream(messages=[]))

        assert len(result) == 2
        # Intermediate chunk: metadata stripped
        assert "model_name" not in chunks[0].generation_info
        assert "id" not in chunks[0].generation_info
        # Final chunk: finish_reason kept
        assert chunks[1].generation_info.get("finish_reason") == "stop"

    @pytest.mark.asyncio
    async def test_uipath_astream_yields_all_chunks_with_metadata_stripping(
        self,
    ) -> None:
        chunks = [
            _make_chunk(
                {"model_name": "gpt-4o", "id": "1", "created": 0}
            ),  # intermediate
            _make_chunk({"finish_reason": "stop"}),  # final
        ]

        async def fake_astream(
            self_inner: Any,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kw: Any,
        ) -> Any:
            for c in chunks:
                yield c

        with patch.dict(
            "os.environ",
            {
                "UIPATH_ACCESS_TOKEN": "eyJhbGciOiAiUlMyNTYiLCAidHlwIjogIkpXVCJ9.eyJzdWIiOiAidGVzdCIsICJleHAiOiA5OTk5OTk5OTk5fQ.fakesignature",
                "UIPATH_URL": "https://dummy.uipath.com",
                "UIPATH_TENANT_ID": "dummy-tenant",
                "UIPATH_ORGANIZATION_ID": "dummy-org",
            },
        ):
            chat = UiPathChat.model_construct()
        with patch.object(_UpstreamUiPathChat, "_uipath_astream", fake_astream):
            result = [c async for c in chat._uipath_astream(messages=[])]

        assert len(result) == 2
        assert "model_name" not in chunks[0].generation_info
        assert chunks[1].generation_info.get("finish_reason") == "stop"
