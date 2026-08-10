"""Tests for extraction_tool.py metadata and functionality."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import httpx
import pytest
from langchain_core.messages import ToolCall
from langgraph.types import Command
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentIxpExtractionResourceConfig,
    AgentIxpExtractionToolProperties,
)
from uipath.eval.mocks._mock_runtime import (
    clear_execution_context,
    set_execution_context,
)
from uipath.eval.mocks._types import (
    LLMMockingStrategy,
    MockingContext,
    ToolSimulation,
)
from uipath.platform.attachments import Attachment
from uipath.platform.documents import ExtractionResponseIXP
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.react.job_attachments import get_job_attachment_paths
from uipath_langchain.agent.react.types import AgentGraphState, InnerAgentGraphState
from uipath_langchain.agent.tools.extraction_tool import (
    StructuredToolWithWrapper,
    create_ixp_extraction_tool,
)
from uipath_langchain.agent.tools.tool_node import AsyncToolWrapperWithState

_ATTACHMENT_ID = "fa93f4ca-bd3f-473a-93e5-e6e5b5a8f27f"

_EXTRACTION_RESPONSE: dict[str, Any] = {
    "extractionResult": {
        "DocumentId": "doc-1",
        "ResultsVersion": 1,
        "ResultsDocument": {},
    },
    "projectId": "proj-1",
    "projectType": "IXP",
    "tag": "v1.0",
    "documentTypeId": "invoice",
    "dataProjection": [
        {
            "fieldGroupName": "invoice",
            "fieldValues": [
                {
                    "id": "total",
                    "name": "Total",
                    "value": "42.00",
                    "unformattedValue": "42.00",
                    "confidence": 0.99,
                    "ocrConfidence": 0.99,
                    "type": "Number",
                }
            ],
        }
    ],
}

_JOB_ATTACHMENT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "attachment": {
            "description": "File to extract data from.",
            "$ref": "#/definitions/job-attachment",
        }
    },
    "required": ["attachment"],
    "definitions": {
        "job-attachment": {
            "type": "object",
            "required": ["ID"],
            "x-uipath-resource-kind": "JobAttachment",
            "properties": {
                "ID": {"type": "string", "description": "Orchestrator attachment key"},
                "FullName": {"type": "string", "description": "File name"},
                "MimeType": {"type": "string", "description": "The MIME type"},
                "Metadata": {
                    "type": "object",
                    "description": "Dictionary<string, string> of metadata",
                    "additionalProperties": {"type": "string"},
                },
            },
        }
    },
}


def _attachment_args(
    *,
    attachment_id: str,
    full_name: str,
    mime_type: str = "application/pdf",
) -> dict[str, Any]:
    """Tool arguments in the shape the job-attachment schema produces."""
    return {
        "attachment": {
            "ID": attachment_id,
            "FullName": full_name,
            "MimeType": mime_type,
        }
    }


def _make_attachment_error(status_code: int, content: bytes) -> EnrichedException:
    req = httpx.Request(
        "GET", "https://cloud.uipath.com/orchestrator_/odata/Attachments(x)"
    )
    resp = httpx.Response(
        status_code,
        request=req,
        content=content,
        headers={"content-type": "application/json"},
    )
    err = httpx.HTTPStatusError(str(status_code), request=req, response=resp)
    enriched = EnrichedException(err)
    enriched.__cause__ = err
    return enriched


class TestExtractionToolMetadata:
    """Test that extraction tool has correct metadata for observability."""

    @pytest.fixture
    def extraction_resource(self):
        """Create a minimal extraction tool resource config."""
        return AgentIxpExtractionResourceConfig(
            name="test_extraction",
            description="Extract data from files",
            input_schema=_JOB_ATTACHMENT_INPUT_SCHEMA,
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProject",
                version_tag="v1.0",
            ),
        )

    def test_extraction_tool_has_correct_name(self, extraction_resource):
        """Test that extraction tool has sanitized name."""
        tool = create_ixp_extraction_tool(extraction_resource)

        assert tool.name == "test_extraction"

    def test_extraction_tool_has_correct_description(self, extraction_resource):
        """Test that extraction tool has correct description."""
        tool = create_ixp_extraction_tool(extraction_resource)

        assert tool.description == "Extract data from files"

    def test_extraction_tool_args_schema_comes_from_the_resource(
        self, extraction_resource
    ):
        """Test that the args schema is generated from the resource input schema."""
        tool = create_ixp_extraction_tool(extraction_resource)

        args_schema = cast(type[BaseModel], tool.args_schema)
        assert tool.metadata is not None
        assert args_schema is tool.metadata["args_schema"]
        assert set(args_schema.model_fields) == {"attachment"}
        assert get_job_attachment_paths(args_schema) == ["$.attachment"]

    @pytest.mark.parametrize(
        "input_schema",
        [
            # `uip agent tool add --type ixp` writes an empty schema
            {"type": "object", "properties": {}},
            {},
        ],
    )
    def test_extraction_tool_falls_back_when_the_resource_schema_is_empty(
        self, input_schema
    ):
        """A resource without a job attachment must still expose one."""
        resource = AgentIxpExtractionResourceConfig(
            name="test_extraction",
            description="Extract data from files",
            input_schema=input_schema,
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProject", version_tag="v1.0"
            ),
        )

        tool = create_ixp_extraction_tool(resource)

        args_schema = cast(type[BaseModel], tool.args_schema)
        assert get_job_attachment_paths(args_schema) == ["$.attachment"]

    def test_extraction_tool_keeps_attachment_id_a_string(self, extraction_resource):
        """Test that the id is not coerced into a UUID object."""
        tool = create_ixp_extraction_tool(extraction_resource)
        attachment_id = "9b702dc7-4988-4fc0-ba81-08deeaade3da"

        parsed = tool._parse_input(
            _attachment_args(attachment_id=attachment_id, full_name="PO_234.pdf"),
            None,
        )

        assert isinstance(parsed, dict)
        sent_id = parsed["attachment"].ID
        assert sent_id == attachment_id
        assert isinstance(sent_id, str)

    def test_extraction_tool_has_extraction_response_output_type(
        self, extraction_resource
    ):
        """Test that extraction tool has ExtractionResponseIXP as output type."""
        tool = create_ixp_extraction_tool(extraction_resource)

        assert hasattr(tool, "output_type")
        assert tool.output_type == ExtractionResponseIXP


class TestExtractionToolFunctionality:
    """Test the extraction tool function behavior."""

    @pytest.fixture
    def extraction_resource(self):
        """Create a minimal extraction tool resource config."""
        return AgentIxpExtractionResourceConfig(
            name="test_extraction",
            description="Extract data from files",
            input_schema=_JOB_ATTACHMENT_INPUT_SCHEMA,
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProject",
                version_tag="v1.0",
            ),
        )

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    @patch("uipath_langchain.agent.tools.extraction_tool.interrupt")
    async def test_extraction_tool_downloads_attachment_and_calls_interrupt(
        self, mock_interrupt, mock_uipath_class, extraction_resource
    ):
        """Test that extraction tool downloads attachment and calls interrupt with correct params."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            return_value="/path/to/document.pdf"
        )
        mock_interrupt.return_value = {"extracted_data": {"field1": "value1"}}

        tool = create_ixp_extraction_tool(extraction_resource)

        result = await tool.ainvoke(
            _attachment_args(
                attachment_id=_ATTACHMENT_ID,
                full_name="document.pdf",
            )
        )

        mock_client.attachments.download_async.assert_called_once_with(
            key=UUID(_ATTACHMENT_ID),
            destination_path="document.pdf",
        )

        assert mock_interrupt.called
        interrupt_arg = mock_interrupt.call_args[0][0]
        assert interrupt_arg.project_name == "TestProject"
        assert interrupt_arg.tag == "v1.0"
        assert interrupt_arg.file_path == "/path/to/document.pdf"

        assert result == {"extracted_data": {"field1": "value1"}}

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    @patch("uipath_langchain.agent.tools.extraction_tool.interrupt")
    async def test_extraction_tool_with_different_version_tag(
        self, mock_interrupt, mock_uipath_class
    ):
        """Test extraction tool with different version tag."""
        extraction_resource = AgentIxpExtractionResourceConfig(
            name="test_extraction_v2",
            description="Extract data from files v2",
            input_schema=_JOB_ATTACHMENT_INPUT_SCHEMA,
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProjectV2",
                version_tag="staging",
            ),
        )

        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            return_value="/path/to/document.pdf"
        )
        mock_interrupt.return_value = {"extracted_data": {}}

        tool = create_ixp_extraction_tool(extraction_resource)

        await tool.ainvoke(
            _attachment_args(
                attachment_id=_ATTACHMENT_ID,
                full_name="document.pdf",
            )
        )

        interrupt_arg = mock_interrupt.call_args[0][0]
        assert interrupt_arg.tag == "staging"

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    async def test_extraction_tool_propagates_download_exception(
        self, mock_uipath_class, extraction_resource
    ):
        """Test that exceptions from attachment download are propagated."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            side_effect=Exception("Download failed")
        )

        tool = create_ixp_extraction_tool(extraction_resource)

        args = _attachment_args(attachment_id=_ATTACHMENT_ID, full_name="file.pdf")

        with pytest.raises(Exception) as exc_info:
            await tool.ainvoke(args)

        assert "Download failed" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    async def test_extraction_tool_permission_denied_raises_deployment(
        self, mock_uipath_class, extraction_resource
    ):
        """A 403/1108 from Orchestrator surfaces as a DEPLOYMENT error, not UNKNOWN."""
        enriched = _make_attachment_error(
            403,
            (
                b'{"message":"You don\'t have permissions to access this attachment.",'
                b'"errorCode":1108}'
            ),
        )

        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(side_effect=enriched)

        tool = create_ixp_extraction_tool(extraction_resource)

        args = _attachment_args(attachment_id=_ATTACHMENT_ID, full_name="file.pdf")

        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.ainvoke(args)

        assert exc_info.value.error_info.category == UiPathErrorCategory.DEPLOYMENT
        assert "permissions" in exc_info.value.error_info.detail

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    async def test_extraction_tool_missing_attachment_raises_system(
        self, mock_uipath_class, extraction_resource
    ):
        """A missing attachment uses the same structured mapping as attachment URI resolution."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            side_effect=_make_attachment_error(
                404,
                b'{"message":"Attachment not found"}',
            )
        )

        tool = create_ixp_extraction_tool(extraction_resource)

        args = _attachment_args(attachment_id=_ATTACHMENT_ID, full_name="file.pdf")

        with pytest.raises(AgentRuntimeError) as exc_info:
            await tool.ainvoke(args)

        assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM
        assert "file.pdf" in exc_info.value.error_info.detail


class TestExtractionToolWrapper:
    """Test the wrapper's job attachment resolution and result handling."""

    @pytest.fixture
    def extraction_resource(self):
        """Create a minimal extraction tool resource config."""
        return AgentIxpExtractionResourceConfig(
            name="test_extraction",
            description="Extract data from files",
            input_schema=_JOB_ATTACHMENT_INPUT_SCHEMA,
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="TestProject",
                version_tag="v1.0",
            ),
        )

    @staticmethod
    def _state(*attachments: Attachment) -> AgentGraphState:
        return AgentGraphState(
            messages=[],
            inner_state=InnerAgentGraphState(
                job_attachments={str(att.id): att for att in attachments}
            ),
        )

    @pytest.mark.asyncio
    @patch("uipath.platform.UiPath")
    @patch("uipath_langchain.agent.tools.extraction_tool.interrupt")
    async def test_wrapper_enriches_the_attachment_from_state(
        self, mock_interrupt, mock_uipath_class, extraction_resource
    ):
        """The model passes only the id; the rest comes from state."""
        mock_client = MagicMock()
        mock_uipath_class.return_value = mock_client
        mock_client.attachments.download_async = AsyncMock(
            return_value="/path/to/document.pdf"
        )
        mock_interrupt.return_value = {"dataProjection": {"total": "42"}}

        tool = cast(
            StructuredToolWithWrapper, create_ixp_extraction_tool(extraction_resource)
        )
        attachment = Attachment(
            id=UUID(_ATTACHMENT_ID),
            full_name="document.pdf",
            mime_type="application/pdf",
        )
        call = {
            "name": tool.name,
            "args": {"attachment": {"ID": _ATTACHMENT_ID}},
            "id": "call-1",
            "type": "tool_call",
        }

        wrapper = cast(AsyncToolWrapperWithState, tool.awrapper)
        result = await wrapper(tool, cast(ToolCall, call), self._state(attachment))

        mock_client.attachments.download_async.assert_called_once_with(
            key=UUID(_ATTACHMENT_ID),
            destination_path="document.pdf",
        )
        assert isinstance(result, Command)
        update = result.update
        assert isinstance(update, dict)
        assert update["messages"][0].content == str({"total": "42"})
        assert update["inner_state"]["tools_storage"] == {
            "test_extraction": {"dataProjection": {"total": "42"}}
        }

    @pytest.mark.asyncio
    async def test_simulated_call_flows_through_the_wrapper(self, extraction_resource):
        """A simulated tool call must reach the wrapper as plain JSON data."""

        async def fake_generate(llm, messages, **kwargs):
            return _EXTRACTION_RESPONSE

        tool = cast(
            StructuredToolWithWrapper, create_ixp_extraction_tool(extraction_resource)
        )
        attachment = Attachment(
            id=UUID(_ATTACHMENT_ID),
            full_name="document.pdf",
            mime_type="application/pdf",
        )
        call = {
            "name": tool.name,
            "args": {"attachment": {"ID": _ATTACHMENT_ID}},
            "id": "call-1",
            "type": "tool_call",
        }

        set_execution_context(
            MockingContext(
                strategy=LLMMockingStrategy(
                    prompt="simulate it",
                    tools_to_simulate=[ToolSimulation(name="test_extraction")],
                ),
                name="run",
                inputs={},
            ),
            MagicMock(),
            "exec-sim",
        )
        try:
            with (
                patch("uipath.eval.mocks._llm_mocker.UiPath", MagicMock()),
                patch(
                    "uipath.eval.mocks._llm_mocker.UiPathLlmChatService", MagicMock()
                ),
                patch(
                    "uipath.eval.mocks._llm_mocker.generate_structured_output",
                    fake_generate,
                ),
                # a simulated tool must not touch the SDK
                patch("uipath.platform.UiPath", MagicMock(side_effect=AssertionError)),
            ):
                wrapper = cast(AsyncToolWrapperWithState, tool.awrapper)
                result = await wrapper(
                    tool, cast(ToolCall, call), self._state(attachment)
                )
        finally:
            clear_execution_context()

        assert isinstance(result, Command)
        update = result.update
        assert isinstance(update, dict)
        # Stored alias-keyed: the shape ixp_escalation_tool re-validates.
        stored = update["inner_state"]["tools_storage"]["test_extraction"]
        assert stored["documentTypeId"] == "invoice"
        assert ExtractionResponseIXP(**stored).data_projection is not None

    @pytest.mark.asyncio
    async def test_wrapper_errors_when_the_attachment_is_not_in_state(
        self, extraction_resource
    ):
        """An id absent from state is reported instead of reaching the SDK."""
        tool = cast(
            StructuredToolWithWrapper, create_ixp_extraction_tool(extraction_resource)
        )
        call = {
            "name": tool.name,
            "args": {"attachment": {"ID": _ATTACHMENT_ID}},
            "id": "call-1",
            "type": "tool_call",
        }

        wrapper = cast(AsyncToolWrapperWithState, tool.awrapper)
        result = await wrapper(tool, cast(ToolCall, call), self._state())

        assert isinstance(result, dict)
        assert _ATTACHMENT_ID in result["error"]


class TestExtractionToolNameSanitization:
    """Test that extraction tool names are properly sanitized."""

    @pytest.mark.asyncio
    async def test_extraction_tool_name_with_spaces(self):
        """Test that tool names with spaces are sanitized."""
        resource = AgentIxpExtractionResourceConfig(
            name="Invoice Extraction Tool",
            description="Extract invoices",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="InvoiceExtraction",
                version_tag="v1.0",
            ),
        )

        tool = create_ixp_extraction_tool(resource)

        assert " " not in tool.name

    @pytest.mark.asyncio
    async def test_extraction_tool_name_with_special_chars(self):
        """Test that tool names with special characters are sanitized."""
        resource = AgentIxpExtractionResourceConfig(
            name="invoice-extraction@v1",
            description="Extract invoices",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentIxpExtractionToolProperties(
                project_name="InvoiceExtraction",
                version_tag="v1.0",
            ),
        )

        tool = create_ixp_extraction_tool(resource)

        # Tool name should be sanitized
        assert tool.name is not None
        assert len(tool.name) > 0
