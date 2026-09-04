"""Tests for the create_output_file internal tool."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from uipath_langchain.agent.tools.internal_tools.output_file_tool import (
    OUTPUT_FILE_TOOL_NAME,
    create_output_file_tool,
    guess_mime_type,
)

ATTACHMENT_ID = "11111111-1111-1111-1111-111111111111"


def args_schema(tool: StructuredTool) -> type[BaseModel]:
    """The tool's argument model, narrowed from the permissive declared union."""
    schema = tool.args_schema
    assert isinstance(schema, type) and issubclass(schema, BaseModel)
    return schema


async def call(tool: StructuredTool, **kwargs: Any) -> dict[str, Any]:
    """Invoke the tool's coroutine directly, bypassing argument validation."""
    coroutine = tool.coroutine
    assert coroutine is not None
    result = await coroutine(**kwargs)
    assert isinstance(result, dict)
    return result


class FakeBackend:
    """Stands in for a filesystem backend with bounded path resolution."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve_path(self, file_path: str) -> Path:
        resolved = (self.root / file_path.lstrip("/")).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"Workspace path escapes root: {file_path}")
        return resolved


@pytest.fixture
def created(monkeypatch) -> list[dict[str, Any]]:
    """Capture every attachment the tool creates."""
    calls: list[dict[str, Any]] = []

    class FakeJobs:
        async def create_attachment_async(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return ATTACHMENT_ID

    class FakeUiPath:
        jobs = FakeJobs()

    monkeypatch.setattr(
        "uipath_langchain.agent.tools.internal_tools.output_file_tool.UiPath",
        lambda *args, **kwargs: FakeUiPath(),
    )
    return calls


class TestGuessMimeType:
    @pytest.mark.parametrize(
        ("file_name", "expected"),
        [
            ("report.md", "text/markdown"),
            ("accounts.csv", "text/csv"),
            ("data.json", "application/json"),
            ("notes.txt", "text/plain"),
            ("config.yaml", "application/yaml"),
            ("book.pdf", "application/pdf"),
            ("mystery", "application/octet-stream"),
            ("REPORT.MD", "text/markdown"),
        ],
    )
    def test_extension_drives_the_mime_type(self, file_name, expected):
        assert guess_mime_type(file_name) == expected


class TestToolSchema:
    def test_content_only_without_a_backend(self):
        properties = args_schema(create_output_file_tool()).model_json_schema()[
            "properties"
        ]

        assert set(properties) == {"file_name", "content"}

    def test_backend_adds_file_path(self, tmp_path):
        tool = create_output_file_tool(FakeBackend(tmp_path))
        properties = args_schema(tool).model_json_schema()["properties"]

        assert set(properties) == {"file_name", "content", "file_path"}

    def test_only_file_name_is_required(self):
        schema = args_schema(create_output_file_tool()).model_json_schema()

        assert schema["required"] == ["file_name"]

    def test_tool_is_named_for_the_prompt(self):
        assert create_output_file_tool().name == OUTPUT_FILE_TOOL_NAME


class TestCreateFromContent:
    async def test_uploads_the_content_and_returns_a_ticket(self, created):
        tool = create_output_file_tool()

        result = await call(tool, file_name="report.md", content="# Report")

        assert result == {
            "file": {
                "ID": ATTACHMENT_ID,
                "FullName": "report.md",
                "MimeType": "text/markdown",
            }
        }
        assert created[0]["name"] == "report.md"
        assert created[0]["content"] == "# Report"
        assert created[0]["source_path"] is None

    async def test_file_name_is_reduced_to_its_basename(self, created):
        tool = create_output_file_tool()

        result = await call(tool, file_name="../../etc/passwd.txt", content="nope")

        assert result["file"]["FullName"] == "passwd.txt"
        assert created[0]["name"] == "passwd.txt"

    async def test_no_source_is_rejected(self, created):
        tool = create_output_file_tool()

        with pytest.raises(ValueError, match="'content'"):
            await call(tool, file_name="report.md")

        assert created == []


class TestCreateFromWorkspacePath:
    async def test_uploads_the_workspace_file(self, created, tmp_path):
        (tmp_path / "report.md").write_text("# Report")
        tool = create_output_file_tool(FakeBackend(tmp_path))

        result = await call(tool, file_name="report.md", file_path="/report.md")

        assert result["file"]["ID"] == ATTACHMENT_ID
        assert created[0]["source_path"] == str(tmp_path / "report.md")
        assert created[0]["content"] is None

    async def test_missing_workspace_file_is_rejected(self, created, tmp_path):
        tool = create_output_file_tool(FakeBackend(tmp_path))

        with pytest.raises(ValueError, match="does not exist in your workspace"):
            await call(tool, file_name="report.md", file_path="/absent.md")

        assert created == []

    async def test_path_escaping_the_workspace_is_rejected(self, created, tmp_path):
        tool = create_output_file_tool(FakeBackend(tmp_path))

        with pytest.raises(ValueError, match="escapes root"):
            await call(tool, file_name="passwd.txt", file_path="../../etc/passwd")

        assert created == []

    async def test_content_and_file_path_together_are_rejected(self, created, tmp_path):
        tool = create_output_file_tool(FakeBackend(tmp_path))

        with pytest.raises(ValueError, match="mutually exclusive"):
            await call(tool, file_name="report.md", content="x", file_path="/report.md")

        assert created == []


class _BackendWithoutPaths:
    """A backend that cannot resolve a workspace path."""


class TestBackendWithoutPathResolution:
    def test_file_path_is_not_offered(self):
        """Advertising it would give the model an argument that always fails."""
        tool = create_output_file_tool(_BackendWithoutPaths())
        properties = args_schema(tool).model_json_schema()["properties"]

        assert set(properties) == {"file_name", "content"}

    async def test_content_still_works(self, created):
        tool = create_output_file_tool(_BackendWithoutPaths())

        result = await call(tool, file_name="report.md", content="# Report")

        assert result["file"]["ID"] == ATTACHMENT_ID
