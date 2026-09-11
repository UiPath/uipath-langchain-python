"""Tests for advanced agent utilities."""

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from uipath_langchain._utils._attachments import (
    ATTACHMENTS_BLOCK_PREFIX,
    render_attachments_block,
)
from uipath_langchain.agent.advanced.types import AdvancedAgentGraphState
from uipath_langchain.agent.advanced.utils import (
    create_state_with_input,
    resolve_input_attachments,
    resolve_message_attachments,
)


class _InputSchema(BaseModel):
    question: str = ""


def test_create_state_returns_base_state_when_schema_is_none() -> None:
    """With no input schema, returns the bare AdvancedAgentGraphState."""
    assert create_state_with_input(None) is AdvancedAgentGraphState


def test_create_state_merges_schema_with_state() -> None:
    """Merged state carries both the base state fields and the input schema fields."""
    merged = create_state_with_input(_InputSchema)
    assert "messages" in merged.model_fields
    assert "structured_response" in merged.model_fields
    assert "question" in merged.model_fields


def test_create_state_does_not_retain_extra_fields() -> None:
    """The combined deep-agent state must not silently allow undeclared keys."""
    state_type = create_state_with_input(_InputSchema)
    state = state_type.model_validate({"question": "value", "undeclared": "extra"})

    assert state_type.model_config.get("extra") != "allow"
    assert "undeclared" not in state.model_dump()


@pytest.mark.asyncio
async def test_resolve_input_attachments_downloads_and_adds_filepath(
    tmp_path: Path,
) -> None:
    """Happy path: download each ticket and augment it with a FilePath."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    attachment_id = uuid.uuid4()
    input_args = {
        "book": {
            "ID": str(attachment_id),
            "FullName": "novel.txt",
            "MimeType": "text/plain",
        },
        "question": "summarize",
    }

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        result = await resolve_input_attachments(backend, ["$.book"], input_args)

    mock_client.attachments.download_async.assert_awaited_once()
    call_kwargs = mock_client.attachments.download_async.call_args.kwargs
    assert call_kwargs["key"] == attachment_id
    expected_name = f"{attachment_id}_novel.txt"
    assert call_kwargs["destination_path"] == str(backend.cwd / expected_name)
    assert result["book"] == {
        "ID": str(attachment_id),
        "FullName": "novel.txt",
        "MimeType": "text/plain",
        "FilePath": f"/{expected_name}",
    }
    assert result["question"] == "summarize"


@pytest.mark.asyncio
async def test_resolve_input_attachments_skips_non_ticket_matches(
    tmp_path: Path,
) -> None:
    """A path match that isn't an attachment ticket is skipped, not downloaded."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    input_args: dict[str, Any] = {"book": "not-a-ticket"}

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        result = await resolve_input_attachments(backend, ["$.book"], input_args)

    mock_client.attachments.download_async.assert_not_awaited()
    assert result == {"book": "not-a-ticket"}


@pytest.mark.asyncio
async def test_resolve_input_attachments_sanitizes_traversal_in_name(
    tmp_path: Path,
) -> None:
    """A traversal-laden FullName is reduced to its basename, staying in the workspace."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    attachment_id = uuid.uuid4()
    input_args: dict[str, Any] = {
        "book": {
            "ID": str(attachment_id),
            "FullName": "../../../etc/passwd",
            "MimeType": "text/plain",
        }
    }

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        result = await resolve_input_attachments(backend, ["$.book"], input_args)

    expected_name = f"{attachment_id}_passwd"
    dest = mock_client.attachments.download_async.call_args.kwargs["destination_path"]
    assert dest == str(backend.cwd / expected_name)
    assert result["book"]["FilePath"] == f"/{expected_name}"


@pytest.mark.asyncio
async def test_resolve_input_attachments_raises_for_non_filesystem_backend() -> None:
    """A backend that isn't FilesystemBackend surfaces a loud NotImplementedError."""
    input_args: dict[str, Any] = {
        "book": {"ID": str(uuid.uuid4()), "FullName": "book.txt"}
    }
    with pytest.raises(NotImplementedError, match="FilesystemBackend"):
        await resolve_input_attachments(None, ["$.book"], input_args)


def _message_with_attachment(attachment_id: uuid.UUID, full_name: str) -> HumanMessage:
    attachments = [
        {"id": str(attachment_id), "full_name": full_name, "mime_type": "text/markdown"}
    ]
    return HumanMessage(
        id="message-1",
        content_blocks=[
            {"type": "text", "text": "can you read this file?"},
            {"type": "text", "text": render_attachments_block(attachments)},
        ],
        additional_kwargs={"attachments": attachments},
    )


@pytest.mark.asyncio
async def test_resolve_message_attachments_downloads_and_adds_file_path(
    tmp_path: Path,
) -> None:
    """A chat attachment lands in the workspace and its path reaches the model."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    attachment_id = uuid.uuid4()
    message = _message_with_attachment(attachment_id, "uipath_company_report.md")

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        updated = await resolve_message_attachments(backend, [message])

    expected_name = f"{attachment_id}_uipath_company_report.md"
    call_kwargs = mock_client.attachments.download_async.call_args.kwargs
    assert call_kwargs["key"] == attachment_id
    assert call_kwargs["destination_path"] == str(backend.cwd / expected_name)

    assert len(updated) == 1
    assert updated[0].id == message.id
    assert updated[0].additional_kwargs["attachments"] == [
        {
            "id": str(attachment_id),
            "full_name": "uipath_company_report.md",
            "mime_type": "text/markdown",
            "file_path": f"/{expected_name}",
        }
    ]
    blocks = [block["text"] for block in updated[0].content]
    assert blocks[0] == "can you read this file?"
    assert f"/{expected_name}" in blocks[1]
    assert blocks[1].count(ATTACHMENTS_BLOCK_PREFIX) == 1


@pytest.mark.asyncio
async def test_resolve_message_attachments_skips_files_already_present(
    tmp_path: Path,
) -> None:
    """Replaying the conversation history on a later exchange downloads nothing."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    attachment_id = uuid.uuid4()
    message = _message_with_attachment(attachment_id, "report.md")
    (backend.cwd / f"{attachment_id}_report.md").write_text("already here")

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        updated = await resolve_message_attachments(backend, [message])

    mock_client.attachments.download_async.assert_not_awaited()
    assert updated[0].additional_kwargs["attachments"][0]["file_path"] == (
        f"/{attachment_id}_report.md"
    )


@pytest.mark.asyncio
async def test_resolve_message_attachments_sanitizes_traversal_in_name(
    tmp_path: Path,
) -> None:
    """A traversal-laden attachment name is reduced to its basename."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    attachment_id = uuid.uuid4()
    message = _message_with_attachment(attachment_id, "../../../etc/passwd")

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        updated = await resolve_message_attachments(backend, [message])

    expected_name = f"{attachment_id}_passwd"
    dest = mock_client.attachments.download_async.call_args.kwargs["destination_path"]
    assert dest == str(backend.cwd / expected_name)
    assert updated[0].additional_kwargs["attachments"][0]["file_path"] == (
        f"/{expected_name}"
    )


@pytest.mark.asyncio
async def test_resolve_message_attachments_leaves_plain_messages_untouched(
    tmp_path: Path,
) -> None:
    """Messages without attachments are neither downloaded nor rewritten."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        updated = await resolve_message_attachments(backend, [HumanMessage("hello")])

    mock_client.attachments.download_async.assert_not_awaited()
    assert updated == []


@pytest.mark.asyncio
async def test_resolve_message_attachments_ignores_non_filesystem_backend() -> None:
    """Without a workspace there is nowhere to download to, so nothing happens."""
    message = _message_with_attachment(uuid.uuid4(), "report.md")
    assert await resolve_message_attachments(None, [message]) == []


@pytest.mark.asyncio
async def test_resolve_message_attachments_survives_a_failed_download(
    tmp_path: Path,
) -> None:
    """One unreachable attachment must not fault the exchange, only lose its path."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    good_id, bad_id = uuid.uuid4(), uuid.uuid4()
    attachments = [
        {"id": str(good_id), "full_name": "good.md", "mime_type": "text/markdown"},
        {"id": str(bad_id), "full_name": "gone.md", "mime_type": "text/markdown"},
    ]
    message = HumanMessage(
        id="message-1",
        content_blocks=[
            {"type": "text", "text": render_attachments_block(attachments)}
        ],
        additional_kwargs={"attachments": attachments},
    )

    async def download(*, key: uuid.UUID, destination_path: str) -> None:
        Path(destination_path).write_bytes(b"")
        if key == bad_id:
            raise RuntimeError("attachment not found")

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock(side_effect=download)
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        updated = await resolve_message_attachments(backend, [message])

    resolved = updated[0].additional_kwargs["attachments"]
    assert resolved[0]["file_path"] == f"/{good_id}_good.md"
    assert "file_path" not in resolved[1]
    assert not (backend.cwd / f"{bad_id}_gone.md").exists()


@pytest.mark.asyncio
async def test_resolve_message_attachments_ignores_non_block_content(
    tmp_path: Path,
) -> None:
    """An assistant message carries plain string content, so there is nothing to path."""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    attachments = [
        {"id": str(uuid.uuid4()), "full_name": "r.md", "mime_type": "text/markdown"}
    ]
    message = AIMessage(
        content="here you go", additional_kwargs={"attachments": attachments}
    )

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        updated = await resolve_message_attachments(backend, [message])

    mock_client.attachments.download_async.assert_not_awaited()
    assert updated == []
