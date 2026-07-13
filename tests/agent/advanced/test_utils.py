"""Tests for advanced agent utilities."""

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends import FilesystemBackend
from pydantic import BaseModel

from uipath_langchain.agent.advanced.types import AdvancedAgentGraphState
from uipath_langchain.agent.advanced.utils import (
    create_state_with_input,
    resolve_input_attachments,
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
async def test_resolve_input_attachments_uses_marked_runtime_workspace_factory(
    tmp_path: Path,
) -> None:
    """Runtime workspace factories resolve from RunnableConfig before download."""
    attachment_id = uuid.uuid4()
    input_args = {
        "book": {
            "ID": str(attachment_id),
            "FullName": "novel.txt",
            "MimeType": "text/plain",
        },
    }

    seen_config: dict[str, Any] = {}

    def backend_factory(runtime: Any) -> FilesystemBackend:
        seen_config.update(runtime.config)
        return FilesystemBackend(
            root_dir=runtime.config["configurable"]["workspace"],
            virtual_mode=True,
        )

    backend_factory.is_uipath_workspace_filesystem_backend = True  # type: ignore[attr-defined]

    mock_client = MagicMock()
    mock_client.attachments.download_async = AsyncMock()
    with patch(
        "uipath_langchain.agent.advanced.utils.UiPath",
        return_value=mock_client,
    ):
        result = await resolve_input_attachments(
            backend_factory,
            ["$.book"],
            input_args,
            config={"configurable": {"workspace": str(tmp_path)}},
        )

    assert seen_config == {"configurable": {"workspace": str(tmp_path)}}
    expected_name = f"{attachment_id}_novel.txt"
    call_kwargs = mock_client.attachments.download_async.call_args.kwargs
    assert call_kwargs["destination_path"] == str(tmp_path / expected_name)
    assert result["book"]["FilePath"] == f"/{expected_name}"


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
