"""Tests for the middleware that downloads tool-produced attachments."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from uipath_langchain.agent.advanced.agent import create_advanced_agent
from uipath_langchain.agent.advanced.tool_attachments import (
    ToolAttachmentsMiddleware,
    find_tickets,
)
from uipath_langchain.agent.attachments.constants import OUTPUT_FILE_TOOL_NAME

_UIPATH = "uipath_langchain.agent.advanced.utils.UiPath"


def _ticket(attachment_id: uuid.UUID, name: str = "report.csv") -> dict[str, Any]:
    return {"ID": str(attachment_id), "FullName": name, "MimeType": "text/csv"}


def _request(tool_name: str = "produce_file") -> MagicMock:
    request = MagicMock()
    request.tool_call = {"name": tool_name, "args": {}, "id": "c1"}
    return request


def _client(*, failing: set[uuid.UUID] = frozenset()) -> MagicMock:
    """A UiPath client whose download writes the destination file."""

    async def download(*, key: uuid.UUID, destination_path: str, **_: Any) -> str:
        if key in failing:
            Path(destination_path).write_bytes(b"partial")
            raise RuntimeError("boom")
        Path(destination_path).write_text("content")
        return destination_path

    client = MagicMock()
    client.attachments.download_async = AsyncMock(side_effect=download)
    return client


async def _passthrough(result: ToolMessage | Command[Any]) -> Any:
    async def handler(_request: Any) -> ToolMessage | Command[Any]:
        return result

    return handler


class TestFindTickets:
    def test_finds_nested_tickets_in_document_order(self) -> None:
        first, second = uuid.uuid4(), uuid.uuid4()
        payload = {
            "result": _ticket(first, "a.csv"),
            "items": [{"nested": {"file": _ticket(second, "b.csv")}}],
        }

        assert [t["ID"] for t in find_tickets(payload)] == [str(first), str(second)]

    def test_skips_objects_that_are_not_tickets(self) -> None:
        payload = {
            "not_uuid": {"ID": "123", "FullName": "a.csv"},
            "no_name": {"ID": str(uuid.uuid4())},
            "empty_name": {"ID": str(uuid.uuid4()), "FullName": ""},
            "plain": "text",
        }

        assert find_tickets(payload) == []

    def test_does_not_search_inside_a_ticket(self) -> None:
        """A ticket's own Metadata is not a place another ticket can hide."""
        inner = uuid.uuid4()
        outer = {**_ticket(uuid.uuid4()), "Metadata": {"file": _ticket(inner)}}

        assert find_tickets({"file": outer}) == [outer]


class TestResolve:
    @pytest.mark.asyncio
    async def test_downloads_and_adds_a_file_path(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        attachment_id = uuid.uuid4()
        message = ToolMessage(
            content=json.dumps({"file": _ticket(attachment_id)}), tool_call_id="c1"
        )
        client = _client()

        with patch(_UIPATH, return_value=client):
            result = await ToolAttachmentsMiddleware(backend).awrap_tool_call(
                _request(), await _passthrough(message)
            )

        expected_name = f"{attachment_id}_report.csv"
        call_kwargs = client.attachments.download_async.call_args.kwargs
        assert call_kwargs["key"] == attachment_id
        assert call_kwargs["destination_path"] == str(tmp_path / expected_name)
        assert isinstance(result, ToolMessage)
        assert json.loads(str(result.content)) == {
            "file": {**_ticket(attachment_id), "FilePath": f"/{expected_name}"}
        }
        assert result.tool_call_id == "c1"

    @pytest.mark.asyncio
    async def test_rewrites_json_inside_text_content_blocks(
        self, tmp_path: Path
    ) -> None:
        """An MCP tool answers with content blocks; the ticket lives in a block's text."""
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        attachment_id = uuid.uuid4()
        message = ToolMessage(
            content=[
                {"type": "text", "text": "prose"},
                {"type": "text", "text": json.dumps(_ticket(attachment_id))},
            ],
            tool_call_id="c1",
        )

        with patch(_UIPATH, return_value=_client()):
            result = await ToolAttachmentsMiddleware(backend).resolve(message)

        assert isinstance(result, ToolMessage)
        blocks = list(result.content)
        assert blocks[0] == {"type": "text", "text": "prose"}
        assert json.loads(blocks[1]["text"])["FilePath"] == (
            f"/{attachment_id}_report.csv"
        )

    @pytest.mark.asyncio
    async def test_rewrites_the_messages_of_a_command(self, tmp_path: Path) -> None:
        """A subagent's ``task`` answer is a Command; its other updates survive."""
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        attachment_id = uuid.uuid4()
        command = Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(_ticket(attachment_id)), tool_call_id="c1"
                    )
                ],
                "todos": ["keep me"],
            },
            goto="somewhere",
        )

        with patch(_UIPATH, return_value=_client()):
            result = await ToolAttachmentsMiddleware(backend).resolve(command)

        assert isinstance(result, Command)
        assert result.goto == "somewhere"
        assert result.update["todos"] == ["keep me"]
        content = json.loads(str(result.update["messages"][0].content))
        assert content["FilePath"] == f"/{attachment_id}_report.csv"

    @pytest.mark.asyncio
    async def test_returns_the_same_object_when_nothing_to_do(
        self, tmp_path: Path
    ) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        client = _client()
        messages = [
            ToolMessage(content="plain prose", tool_call_id="c1"),
            ToolMessage(content=json.dumps({"rows": 3}), tool_call_id="c2"),
            ToolMessage(
                content=json.dumps(_ticket(uuid.uuid4())),
                tool_call_id="c3",
                status="error",
            ),
        ]

        with patch(_UIPATH, return_value=client):
            for message in messages:
                assert (
                    await ToolAttachmentsMiddleware(backend).resolve(message) is message
                )

        client.attachments.download_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_the_output_file_tool(self, tmp_path: Path) -> None:
        """The agent wrote that file itself; fetching a copy back is waste."""
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        message = ToolMessage(
            content=json.dumps({"file": _ticket(uuid.uuid4())}), tool_call_id="c1"
        )
        client = _client()

        with patch(_UIPATH, return_value=client):
            result = await ToolAttachmentsMiddleware(backend).awrap_tool_call(
                _request(OUTPUT_FILE_TOOL_NAME), await _passthrough(message)
            )

        assert result is message
        client.attachments.download_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_fetch_a_file_already_in_the_workspace(
        self, tmp_path: Path
    ) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        attachment_id = uuid.uuid4()
        (tmp_path / f"{attachment_id}_report.csv").write_text("already here")
        message = ToolMessage(
            content=json.dumps(_ticket(attachment_id)), tool_call_id="c1"
        )
        client = _client()

        with patch(_UIPATH, return_value=client):
            result = await ToolAttachmentsMiddleware(backend).resolve(message)

        client.attachments.download_async.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert json.loads(str(result.content))["FilePath"] == (
            f"/{attachment_id}_report.csv"
        )

    @pytest.mark.asyncio
    async def test_a_failed_download_leaves_that_ticket_without_a_path(
        self, tmp_path: Path
    ) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        good, bad = uuid.uuid4(), uuid.uuid4()
        message = ToolMessage(
            content=json.dumps(
                {
                    "good": _ticket(good, "good.csv"),
                    "bad": {**_ticket(bad, "bad.csv"), "FilePath": "/stale"},
                }
            ),
            tool_call_id="c1",
        )

        with patch(_UIPATH, return_value=_client(failing={bad})):
            result = await ToolAttachmentsMiddleware(backend).resolve(message)

        assert isinstance(result, ToolMessage)
        content = json.loads(str(result.content))
        assert content["good"]["FilePath"] == f"/{good}_good.csv"
        assert "FilePath" not in content["bad"]
        assert not (tmp_path / f"{bad}_bad.csv").exists()

    def test_sync_tool_calls_download_too(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
        attachment_id = uuid.uuid4()
        message = ToolMessage(
            content=json.dumps(_ticket(attachment_id)), tool_call_id="c1"
        )

        with patch(_UIPATH, return_value=_client()):
            result = ToolAttachmentsMiddleware(backend).wrap_tool_call(
                _request(), lambda _request: message
            )

        assert isinstance(result, ToolMessage)
        assert json.loads(str(result.content))["FilePath"] == (
            f"/{attachment_id}_report.csv"
        )


# --- End to end on a real deep agent ---

_ATTACHMENT_ID = uuid.uuid4()
_MODEL_INPUTS: list[list[BaseMessage]] = []


class _RecordingModel(GenericFakeChatModel):
    """Records the messages of every model call and accepts any tool binding."""

    model_name: str = "test-model-tool-attachments"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return {"ls_provider": "openai", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_RecordingModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        _MODEL_INPUTS.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _produce_file_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda name="report.csv": {"file": _ticket(_ATTACHMENT_ID, name)},
        name="produce_file",
        description="produce a file",
    )


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def _file_path_tool_messages() -> list[ToolMessage]:
    """Every ToolMessage any agent was shown that carries the downloaded path."""
    expected = f"/{_ATTACHMENT_ID}_report.csv"
    return [
        message
        for messages in _MODEL_INPUTS
        for message in messages
        if isinstance(message, ToolMessage)
        and message.name == "produce_file"
        and expected in str(message.content)
    ]


def _run(tmp_path: Path, scripted: list[AIMessage]) -> dict[str, Any]:
    _MODEL_INPUTS.clear()
    model = _RecordingModel(
        messages=iter([*scripted, *[AIMessage(content="done")] * 20])
    )
    graph = create_advanced_agent(
        model=model,
        tools=[_produce_file_tool()],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )
    with patch(_UIPATH, return_value=_client()):
        return asyncio.run(
            graph.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
        )


def test_the_main_agent_sees_the_path_of_a_tool_produced_file(tmp_path: Path) -> None:
    result = _run(tmp_path, [_tool_call("produce_file", {"name": "report.csv"}, "c1")])

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert json.loads(str(tool_messages[0].content))["file"]["FilePath"] == (
        f"/{_ATTACHMENT_ID}_report.csv"
    )
    assert (tmp_path / f"{_ATTACHMENT_ID}_report.csv").read_text() == "content"


def test_a_subagent_sees_the_path_of_a_tool_produced_file(tmp_path: Path) -> None:
    """The subagent shares the workspace, so its own tool results get a path too."""
    _run(
        tmp_path,
        [
            _tool_call(
                "task", {"description": "go", "subagent_type": "general-purpose"}, "c1"
            ),
            _tool_call("produce_file", {"name": "report.csv"}, "c2"),
        ],
    )

    assert _file_path_tool_messages(), "no agent was shown the downloaded path"
    assert (tmp_path / f"{_ATTACHMENT_ID}_report.csv").read_text() == "content"
