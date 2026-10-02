"""Contract test: attachment references in an advanced agent's tool calls.

Runs a real deepagents graph, since the behaviour under test is how LangChain's
ToolNode and deepagents' subagent dispatch treat the middleware and its state.
"""

import asyncio
from pathlib import Path
from typing import Any, Sequence

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from uipath_langchain.agent.advanced.agent import create_advanced_agent
from uipath_langchain.agent.advanced.job_attachments_middleware import (
    JOB_ATTACHMENTS_STATE_KEY,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    JOB_ATTACHMENT_DEFINITION,
    single_attachment_schema,
)
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)

from ..attachments.fake_orchestrator import patch_orchestrator

KNOWN_ID = "55555555-5555-5555-5555-555555555555"
CHILD_ID = "66666666-6666-6666-6666-666666666666"
INVENTED_ID = "77777777-7777-7777-7777-777777777777"
KNOWN = {"ID": KNOWN_ID, "FullName": "known.md", "MimeType": "text/markdown"}


class _Model(GenericFakeChatModel):
    model_name: str = "test-model-attachments"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return {"ls_provider": "openai", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_Model":
        return self


def _reader(received: list[dict[str, Any]]) -> BaseTool:
    """A tool that takes a file and records the argument it was given."""

    def read(**kwargs: Any) -> str:
        received.append(kwargs["document"].model_dump(exclude_none=True))
        return "read"

    return StructuredTool(
        name="read_document",
        description="Read a document.",
        args_schema=create_model(
            {
                "type": "object",
                "properties": {"document": {"$ref": "#/definitions/job-attachment"}},
                "required": ["document"],
                "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
            }
        ),
        func=read,
    )


def _producer() -> BaseTool:
    """A tool that returns a file, the way a process tool does."""
    output_model = create_model(single_attachment_schema("file", "The file."))
    return StructuredToolWithOutputType(
        name="produce_file",
        description="Produce a file.",
        args_schema=create_model({"type": "object", "properties": {}}),
        func=lambda **_: (
            '{"file": {"ID": "%s", "FullName": "child.md", "MimeType": "text/markdown"}}'
            % CHILD_ID
        ),
        output_type=output_model,
    )


def _call(name: str, args: dict[str, Any], id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def _run(
    tmp_path: Path,
    tools: list[BaseTool],
    turns: list[AIMessage],
    known: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = _Model(messages=iter([*turns, *[AIMessage(content="done")] * 10]))
    graph = create_advanced_agent(
        model=model,
        tools=tools,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )
    state: dict[str, Any] = {"messages": [{"role": "user", "content": "go"}]}
    if known is not None:
        state[JOB_ATTACHMENTS_STATE_KEY] = known
    return asyncio.run(graph.ainvoke(state))


def _tool_messages(result: dict[str, Any], name: str) -> list[ToolMessage]:
    return [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == name
    ]


def test_a_known_reference_reaches_the_tool_as_the_stored_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    received: list[dict[str, Any]] = []

    _run(
        tmp_path,
        [_reader(received)],
        [_call("read_document", {"document": {"ID": KNOWN_ID}}, "c1")],
        known={KNOWN_ID: KNOWN},
    )

    assert received == [KNOWN]


def test_an_invented_reference_is_rejected_without_running_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_orchestrator(monkeypatch, existing={})
    received: list[dict[str, Any]] = []

    result = _run(
        tmp_path,
        [_reader(received)],
        [_call("read_document", {"document": {"ID": INVENTED_ID}}, "c1")],
    )

    (message,) = _tool_messages(result, "read_document")
    assert message.status == "error"
    assert INVENTED_ID in str(message.content)
    assert received == []


def test_a_file_a_tool_returned_is_remembered_and_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    received: list[dict[str, Any]] = []

    result = _run(
        tmp_path,
        [_producer(), _reader(received)],
        [
            _call("produce_file", {}, "c1"),
            _call("read_document", {"document": {"ID": CHILD_ID}}, "c2"),
        ],
    )

    assert CHILD_ID in result[JOB_ATTACHMENTS_STATE_KEY]
    assert received[0]["FullName"] == "child.md"


def test_a_file_from_another_job_is_found_in_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = patch_orchestrator(monkeypatch, existing={CHILD_ID: "rpa.md"})
    received: list[dict[str, Any]] = []

    result = _run(
        tmp_path,
        [_reader(received)],
        [_call("read_document", {"document": {"ID": CHILD_ID}}, "c1")],
    )

    assert received[0]["FullName"] == "rpa.md"
    assert fake.lookups == [CHILD_ID]
    assert CHILD_ID in result[JOB_ATTACHMENTS_STATE_KEY]


def test_a_subagent_sees_the_parents_attachments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """deepagents copies non-private state keys into the subagent it dispatches."""
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    received: list[dict[str, Any]] = []

    _run(
        tmp_path,
        [_reader(received)],
        [
            _call(
                "task",
                {"description": "read it", "subagent_type": "general-purpose"},
                "c1",
            ),
            _call("read_document", {"document": {"ID": KNOWN_ID}}, "c2"),
        ],
        known={KNOWN_ID: KNOWN},
    )

    assert received == [KNOWN]
