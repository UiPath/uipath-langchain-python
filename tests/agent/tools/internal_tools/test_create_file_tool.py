"""Tests for the create-file internal tool."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel
from uipath.agent.models.agent import AgentInternalToolResourceConfig

from uipath_langchain.agent.attachments.mime_types import guess_mime_type
from uipath_langchain.agent.tools.internal_tools.create_file_tool import (
    CreateFileTool,
    create_file_tool,
)
from uipath_langchain.agent.tools.internal_tools.internal_tool_factory import (
    create_internal_tool,
)
from uipath_langchain.agent.tools.static_args import StaticArgsHandler

ATTACHMENT_ID = "11111111-1111-1111-1111-111111111111"


STANDARD_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fileName": {"type": "string", "description": "Saved file name text."},
        "content": {"type": "string", "description": "Saved content text."},
        "filePath": {"type": "string", "description": "Never set this, use content."},
    },
    "required": ["fileName"],
}

SAVED_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fileName": {"type": "string", "description": "Saved file name text."},
        "content": {"type": "string", "description": "Saved content text."},
        "filePath": {"type": "string", "description": "Saved file path text."},
    },
    "required": ["fileName"],
}


def create_file_resource(
    name: str = "Create File",
    description: str = "Make a file.",
    input_schema: dict[str, Any] = SAVED_INPUT_SCHEMA,
    argument_properties: dict[str, Any] | None = None,
) -> AgentInternalToolResourceConfig:
    return AgentInternalToolResourceConfig.model_validate(
        {
            "$resourceType": "tool",
            "type": "Internal",
            "name": name,
            "description": description,
            "properties": {"toolType": "create-file"},
            "inputSchema": input_schema,
            "argumentProperties": argument_properties or {},
        }
    )


class _NoInput(BaseModel):
    pass


def make_tool(backend: Any | None = None) -> CreateFileTool:
    return create_file_tool(create_file_resource(), backend)


def args_schema(tool: StructuredTool) -> type[BaseModel]:
    """The tool's argument model, narrowed from the permissive declared union."""
    schema = tool.args_schema
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    return schema


async def call(tool: StructuredTool, **kwargs: Any) -> dict[str, Any]:
    """Invoke the tool's coroutine directly, bypassing argument validation."""
    coroutine = tool.coroutine
    assert coroutine is not None
    result = await coroutine(**kwargs)
    assert isinstance(result, dict)
    return result


class FakeBackend:
    """Stands in for a backend that exposes a workspace root."""

    def __init__(self, root: Path) -> None:
        self.cwd = root.resolve()


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
        "uipath_langchain.agent.tools.internal_tools.create_file_tool.UiPath",
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

    def test_csv_ignores_the_windows_registry_mapping(self, monkeypatch):
        monkeypatch.setattr(
            "mimetypes.guess_type",
            lambda *_args, **_kwargs: ("application/vnd.ms-excel", None),
        )

        assert guess_mime_type("accounts.csv") == "text/csv"


class TestToolSchema:
    def test_output_schema_is_the_shared_single_attachment_shape(self):
        from uipath_langchain.agent.tools.internal_tools.create_file_tool import (
            create_file_tool_output_schema,
        )
        from uipath_langchain.agent.tools.internal_tools.schema_utils import (
            single_attachment_schema,
        )

        schema = create_file_tool_output_schema()

        assert schema["required"] == ["file"]
        assert schema == single_attachment_schema(
            "file", schema["properties"]["file"]["description"]
        )

    @pytest.mark.parametrize(
        "saved",
        [STANDARD_INPUT_SCHEMA, SAVED_INPUT_SCHEMA],
        ids=["standard", "advanced"],
    )
    def test_the_saved_schema_is_what_the_model_sees(self, saved):
        tool = create_file_tool(create_file_resource(input_schema=saved))
        schema = args_schema(tool).model_json_schema()

        assert set(schema["properties"]) == set(saved["properties"])
        assert schema["required"] == saved["required"]

    def test_saved_descriptions_reach_the_model(self, tmp_path):
        tool = make_tool(FakeBackend(tmp_path))
        properties = args_schema(tool).model_json_schema()["properties"]

        assert {name: p["description"] for name, p in properties.items()} == {
            "fileName": "Saved file name text.",
            "content": "Saved content text.",
            "filePath": "Saved file path text.",
        }

    def test_name_and_description_come_from_the_resource(self):
        tool = create_file_tool(
            create_file_resource(name="Make Report", description="Writes it.")
        )

        assert tool.name == "Make_Report"
        assert tool.description == "Writes it."

    def test_factory_builds_the_tool(self):
        tool = create_internal_tool(create_file_resource(), llm=None)  # type: ignore[arg-type]

        assert isinstance(tool, CreateFileTool)

    def test_a_static_argument_is_pinned_and_filled_in(self):
        tool = create_file_tool(
            create_file_resource(
                input_schema=STANDARD_INPUT_SCHEMA,
                argument_properties={
                    "$['fileName']": {
                        "variant": "static",
                        "value": "report.md",
                        "isSensitive": False,
                    }
                },
            )
        )
        handler = StaticArgsHandler()
        (bound,) = handler.initialize([tool], _NoInput(), _NoInput)
        call = ToolCall(
            name=tool.name, args={"content": "# Report"}, id="1", type="tool_call"
        )

        handler.apply_to_response([call])

        assert isinstance(bound, StructuredTool)
        schema = args_schema(bound).model_json_schema()
        ref = schema["properties"]["fileName"]["$ref"].rsplit("/", 1)[-1]
        pinned = schema["$defs"][ref]
        assert pinned["enum"] == ["report.md"]
        assert call["args"] == {"content": "# Report", "fileName": "report.md"}


class TestWithWorkspace:
    def test_offers_file_path_and_keeps_metadata(self, tmp_path):
        tool = make_tool()
        assert tool.metadata is not None
        tool.metadata["tool_id"] = "node-1"

        rebound = tool.with_workspace(FakeBackend(tmp_path))

        assert "filePath" in args_schema(rebound).model_json_schema()["properties"]
        assert rebound.metadata is not None
        assert rebound.metadata["tool_id"] == "node-1"
        assert rebound.metadata["args_schema"] is rebound.args_schema
        assert rebound.name == tool.name


class TestCreateFromContent:
    async def test_uploads_the_content_and_returns_a_ticket(self, created):
        tool = make_tool()

        result = await call(tool, fileName="report.md", content="# Report")

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
        tool = make_tool()

        result = await call(tool, fileName="../../etc/passwd.txt", content="nope")

        assert result["file"]["FullName"] == "passwd.txt"
        assert created[0]["name"] == "passwd.txt"

    async def test_no_source_is_rejected(self, created):
        tool = make_tool()

        with pytest.raises(ToolException, match="'content'"):
            await call(tool, fileName="report.md")

        assert created == []

    async def test_file_path_without_a_workspace_is_rejected(self, created):
        tool = make_tool()

        with pytest.raises(ToolException):
            await call(tool, fileName="report.md", filePath="/report.md")

        assert created == []


class TestErrorsGoBackToTheModel:
    async def test_a_bad_call_returns_an_error_result_instead_of_raising(self, created):
        tool = make_tool()

        message = await tool.ainvoke(
            ToolCall(
                name=tool.name,
                args={"fileName": "report.md"},
                id="call-1",
                type="tool_call",
            )
        )

        assert message.status == "error"
        assert "'content'" in message.content
        assert created == []


class TestCreateFromWorkspacePath:
    async def test_uploads_the_workspace_file(self, created, tmp_path):
        (tmp_path / "report.md").write_text("# Report")
        tool = make_tool(FakeBackend(tmp_path))

        result = await call(tool, fileName="report.md", filePath="/report.md")

        assert result["file"]["ID"] == ATTACHMENT_ID
        assert created[0]["source_path"] == str(tmp_path / "report.md")
        assert created[0]["content"] is None

    async def test_missing_workspace_file_is_rejected(self, created, tmp_path):
        tool = make_tool(FakeBackend(tmp_path))

        with pytest.raises(ToolException, match="does not exist in your workspace"):
            await call(tool, fileName="report.md", filePath="/absent.md")

        assert created == []

    @pytest.mark.parametrize(
        "file_path", ["../../etc/passwd", "/../outside.txt", "/sub/../../escape.txt"]
    )
    async def test_traversal_is_rejected(self, created, tmp_path, file_path):
        tool = make_tool(FakeBackend(tmp_path))

        with pytest.raises(ToolException, match="traversal"):
            await call(tool, fileName="x.txt", filePath=file_path)

        assert created == []

    async def test_symlink_out_of_the_workspace_is_rejected(self, created, tmp_path):
        """The marker check cannot see this one; containment after resolve can."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("x")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "link.txt").symlink_to(outside / "secret.txt")
        tool = make_tool(FakeBackend(workspace))

        with pytest.raises(ToolException, match="outside your workspace"):
            await call(tool, fileName="secret.txt", filePath="/link.txt")

        assert created == []

    async def test_a_relative_path_is_read_from_the_workspace_root(
        self, created, tmp_path
    ):
        (tmp_path / "report.md").write_text("# Report")
        tool = make_tool(FakeBackend(tmp_path))

        await call(tool, fileName="report.md", filePath="report.md")

        assert created[0]["source_path"] == str((tmp_path / "report.md").resolve())

    async def test_content_and_file_path_together_are_rejected(self, created, tmp_path):
        tool = make_tool(FakeBackend(tmp_path))

        with pytest.raises(ToolException, match="mutually exclusive"):
            await call(tool, fileName="report.md", content="x", filePath="/report.md")

        assert created == []


class _BackendWithoutPaths:
    """A backend that exposes no workspace root."""


class TestBackendWithoutPathResolution:
    async def test_file_path_is_rejected(self, created):
        tool = make_tool(_BackendWithoutPaths())

        with pytest.raises(ToolException):
            await call(tool, fileName="report.md", filePath="/report.md")

        assert created == []

    async def test_content_still_works(self, created):
        tool = make_tool(_BackendWithoutPaths())

        result = await call(tool, fileName="report.md", content="# Report")

        assert result["file"]["ID"] == ATTACHMENT_ID
