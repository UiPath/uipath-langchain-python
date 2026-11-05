"""Tests for the create-file internal tool."""

import asyncio
import gzip
import io
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel
from pypdf import PdfReader
from uipath.agent.models.agent import AgentInternalToolResourceConfig

import uipath_langchain.agent.tools.internal_tools as internal_tools_package
import uipath_langchain.agent.tools.internal_tools.create_file_tool as create_file_module
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


PDF_RENDERER_MODULE = "uipath_langchain.agent.tools.internal_tools.pdf_renderer"


class Renderers:
    def __init__(self) -> None:
        self.processes: list[asyncio.subprocess.Process] = []
        self.spawned = asyncio.Event()


@pytest.fixture
def renderers(monkeypatch: pytest.MonkeyPatch) -> Renderers:
    seen = Renderers()
    spawn = asyncio.create_subprocess_exec

    async def spawn_and_record(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await spawn(*args, **kwargs)
        seen.processes.append(process)
        seen.spawned.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_and_record)
    return seen


def uploaded_pdf_text(upload: dict[str, Any]) -> str:
    content = upload["content"]
    assert isinstance(content, bytes)
    assert content.startswith(b"%PDF-")
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() for page in reader.pages)


class TestCreatePdf:
    async def test_markdown_content_is_rendered_into_a_pdf(self, created, pdf_fonts):
        tool = make_tool()

        result = await call(
            tool, fileName="report.pdf", content="# Quarterly report\n\nRevenue up."
        )

        assert result == {
            "file": {
                "ID": ATTACHMENT_ID,
                "FullName": "report.pdf",
                "MimeType": "application/pdf",
            }
        }
        text = uploaded_pdf_text(created[0])
        assert "Quarterly report" in text
        assert "#" not in text
        assert b"TestSans-Regular" in created[0]["content"]
        assert created[0]["source_path"] is None

    async def test_the_extension_is_matched_case_insensitively(
        self, created, pdf_fonts
    ):
        tool = make_tool()

        await call(tool, fileName="REPORT.PDF", content="Revenue up.")

        assert "Revenue up." in uploaded_pdf_text(created[0])

    async def test_only_the_last_extension_decides(self, created):
        tool = make_tool()

        await call(tool, fileName="report.pdf.md", content="# Report")

        assert created[0]["content"] == "# Report"

    async def test_characters_without_a_font_are_logged(
        self, created, pdf_fonts, caplog
    ):
        tool = make_tool()

        with caplog.at_level(logging.WARNING, logger="uipath"):
            result = await call(tool, fileName="report.pdf", content="Thai ก, done 🚀")

        assert "Thai" in uploaded_pdf_text(created[0])
        assert result == {
            "file": {
                "ID": ATTACHMENT_ID,
                "FullName": "report.pdf",
                "MimeType": "application/pdf",
            }
        }
        (record,) = [r for r in caplog.records if "report.pdf" in r.getMessage()]
        assert record.name == "uipath"
        assert "2 characters" in record.getMessage()
        assert "'ก'" in record.getMessage()
        assert "'🚀'" in record.getMessage()

    async def test_logged_missing_characters_are_capped(
        self, created, pdf_fonts, caplog
    ):
        uncovered = "".join(chr(0x10000 + i) for i in range(200))
        tool = make_tool()

        with caplog.at_level(logging.WARNING, logger="uipath"):
            await call(tool, fileName="report.pdf", content=uncovered)

        (record,) = [r for r in caplog.records if "report.pdf" in r.getMessage()]
        assert "200 characters" in record.getMessage()
        assert repr(chr(0x10000 + 49)) in record.getMessage()
        assert repr(chr(0x10000 + 50)) not in record.getMessage()

    async def test_content_over_the_size_limit_is_rejected(self, created, monkeypatch):
        monkeypatch.setattr(create_file_module, "_MAX_PDF_MARKDOWN_CHARS", 10)
        tool = make_tool()

        with pytest.raises(ToolException, match="11 characters"):
            await call(tool, fileName="report.pdf", content="x" * 11)

        assert created == []

    async def test_content_at_the_size_limit_is_rendered(
        self, created, pdf_fonts, monkeypatch
    ):
        monkeypatch.setattr(create_file_module, "_MAX_PDF_MARKDOWN_CHARS", 10)
        tool = make_tool()

        await call(tool, fileName="report.pdf", content="x" * 10)

        assert "x" * 10 in uploaded_pdf_text(created[0])

    async def test_a_workspace_file_of_four_byte_characters_at_the_limit_is_rendered(
        self, created, pdf_fonts, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(create_file_module, "_MAX_PDF_MARKDOWN_CHARS", 10)
        (tmp_path / "report.md").write_text("𝐀" * 10, "utf-8")
        tool = make_tool(FakeBackend(tmp_path))

        await call(tool, fileName="report.pdf", filePath="/report.md")

        assert created[0]["content"].startswith(b"%PDF-")

    async def test_a_workspace_file_over_the_size_limit_is_rejected(
        self, created, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(create_file_module, "_MAX_PDF_MARKDOWN_CHARS", 10)
        (tmp_path / "report.md").write_text("x" * 11, "utf-8")
        tool = make_tool(FakeBackend(tmp_path))

        with pytest.raises(ToolException, match="11 characters"):
            await call(tool, fileName="report.pdf", filePath="/report.md")

        assert created == []

    async def test_a_huge_workspace_file_is_rejected_before_it_is_read(
        self, created, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(create_file_module, "_MAX_PDF_MARKDOWN_CHARS", 10)
        (tmp_path / "dump.log").write_bytes(b"\xff" * 41)
        tool = make_tool(FakeBackend(tmp_path))

        with pytest.raises(ToolException, match="41 bytes"):
            await call(tool, fileName="dump.pdf", filePath="/dump.log")

        assert created == []

    async def test_rendering_runs_outside_the_agent_process(
        self, created, pdf_fonts, monkeypatch
    ):
        pdf_renderer = pytest.importorskip(PDF_RENDERER_MODULE)

        def fail_in_this_process(_markdown: str) -> None:
            raise AssertionError("rendered inside the agent process")

        monkeypatch.setattr(pdf_renderer, "render_markdown_pdf", fail_in_this_process)
        tool = make_tool()

        await call(tool, fileName="report.pdf", content="Revenue up.")

        assert "Revenue up." in uploaded_pdf_text(created[0])

    async def test_a_render_that_takes_too_long_is_stopped(
        self, created, pdf_fonts, monkeypatch, renderers
    ):
        monkeypatch.setattr(create_file_module, "_PDF_RENDER_TIMEOUT_SECONDS", 0.01)
        tool = make_tool()

        with pytest.raises(ToolException, match="took longer than 0.01 seconds"):
            await call(tool, fileName="report.pdf", content="Revenue up.")

        assert renderers.processes[0].returncode is not None
        assert created == []

    async def test_a_cancelled_render_stops_the_renderer(
        self, created, pdf_fonts, renderers
    ):
        tool = make_tool()
        task = asyncio.create_task(
            call(tool, fileName="report.pdf", content="Revenue up.")
        )
        await asyncio.wait_for(renderers.spawned.wait(), timeout=10)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert renderers.processes[0].returncode is not None
        assert created == []

    async def test_a_renderer_that_cannot_start_is_reported_to_the_model(
        self, created, pdf_fonts, monkeypatch, tmp_path, caplog
    ):
        monkeypatch.setattr(sys, "executable", str(tmp_path / "missing-python"))
        tool = make_tool()

        with caplog.at_level(logging.WARNING, logger="uipath"):
            with pytest.raises(ToolException, match="could not be rendered"):
                await call(tool, fileName="report.pdf", content="Revenue up.")

        missing_python = str(tmp_path / "missing-python")
        assert any(missing_python in r.getMessage() for r in caplog.records)
        assert created == []

    async def test_a_renderer_crash_is_logged_and_reported_to_the_model(
        self, created, pdf_fonts, caplog
    ):
        (pdf_fonts / "LiberationSans-Regular.ttf").write_bytes(b"not a font")
        tool = make_tool()

        with caplog.at_level(logging.WARNING, logger="uipath"):
            with pytest.raises(ToolException, match="could not be rendered"):
                await call(tool, fileName="report.pdf", content="Revenue up.")

        (record,) = [r for r in caplog.records if "exited with code" in r.getMessage()]
        assert record.name == "uipath"
        assert "TTLibError" in record.getMessage()
        assert created == []

    async def test_a_pdf_over_the_size_limit_is_rejected(
        self, created, pdf_fonts, monkeypatch
    ):
        monkeypatch.setattr(create_file_module, "_MAX_PDF_BYTES", 100)
        tool = make_tool()

        with pytest.raises(ToolException, match="larger than 100 bytes"):
            await call(tool, fileName="report.pdf", content="Revenue up.")

        assert created == []

    @pytest.mark.skipif(sys.platform == "win32", reason="memory limits need POSIX")
    async def test_a_render_that_runs_out_of_memory_is_rejected(
        self, created, pdf_fonts, monkeypatch
    ):
        monkeypatch.setattr(
            create_file_module, "_MAX_PDF_RENDER_MEMORY_BYTES", 128 * 1024 * 1024
        )
        tool = make_tool()

        await call(tool, fileName="small.pdf", content="Revenue up.")
        with pytest.raises(ToolException, match="more memory than 128 MB"):
            await call(
                tool, fileName="report.pdf", content="<http://a/" + "報" * 20000 + ">"
            )

        assert [upload["name"] for upload in created] == ["small.pdf"]

    async def test_a_workspace_file_that_is_neither_pdf_nor_text_is_rejected(
        self, created, tmp_path
    ):
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
        tool = make_tool(FakeBackend(tmp_path))

        with pytest.raises(ToolException, match="neither a PDF nor UTF-8 text"):
            await call(tool, fileName="image.pdf", filePath="/image.png")

        assert created == []

    async def test_a_workspace_pdf_is_uploaded_unchanged(self, created, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "scan.pdf").write_bytes(b"%PDF-1.7\nexisting")
        tool = make_tool(FakeBackend(workspace))

        await call(tool, fileName="scan.pdf", filePath="/scan.pdf")

        assert created[0]["source_path"] == str(workspace / "scan.pdf")
        assert created[0]["content"] is None

    async def test_a_compressed_workspace_pdf_is_uploaded_unchanged(
        self, created, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "scan.pdf.gz").write_bytes(gzip.compress(b"%PDF-1.7\nexisting"))
        tool = make_tool(FakeBackend(workspace))

        await call(tool, fileName="scan.pdf.gz", filePath="/scan.pdf.gz")

        assert created[0]["source_path"] == str(workspace / "scan.pdf.gz")
        assert created[0]["content"] is None

    async def test_a_text_file_that_mentions_the_pdf_header_is_rendered(
        self, created, pdf_fonts, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "notes.pdf").write_text("Files start with %PDF-1.7.", "utf-8")
        tool = make_tool(FakeBackend(workspace))

        await call(tool, fileName="notes.pdf", filePath="/notes.pdf")

        assert created[0]["source_path"] is None
        assert "Files start with" in uploaded_pdf_text(created[0])

    async def test_a_workspace_text_file_named_pdf_is_rendered(
        self, created, pdf_fonts, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "report.md").write_text("# Summary\n\nAll good.", "utf-8")
        tool = make_tool(FakeBackend(workspace))

        await call(tool, fileName="report.pdf", filePath="/report.md")

        assert "All good." in uploaded_pdf_text(created[0])
        assert created[0]["source_path"] is None

    async def test_without_the_pdf_extra_the_model_is_told_how_to_install_it(
        self, created, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "reportlab", None)
        for name in [*sys.modules]:
            if name.startswith("reportlab."):
                monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, PDF_RENDERER_MODULE, raising=False)
        monkeypatch.delattr(internal_tools_package, "pdf_renderer", raising=False)
        tool = make_tool()

        with pytest.raises(ToolException, match=r"uipath-langchain\[pdf\]"):
            await call(tool, fileName="report.pdf", content="Revenue up.")

        assert created == []
