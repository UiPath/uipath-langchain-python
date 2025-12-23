"""Tests for job_attachment_wrapper module."""

import uuid
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.wrappers.job_attachment_wrapper import (
    get_job_attachment_wrapper,
)


class MockAttachmentSchema(BaseModel):
    """Mock schema with job attachment field."""

    attachment_id: uuid.UUID = Field(description="Job attachment ID")
    name: str


class TestGetJobAttachmentWrapper:
    """Test cases for get_job_attachment_wrapper function."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = MagicMock(spec=BaseTool)
        tool.ainvoke = AsyncMock(return_value={"result": "success"})
        return tool

    @pytest.fixture
    def mock_tool_call(self):
        """Create a mock tool call."""
        return {
            "name": "test_tool",
            "args": {"attachment_id": str(uuid.uuid4()), "name": "test"},
            "id": "call_123",
        }

    @pytest.fixture
    def mock_state(self):
        """Create a mock agent graph state."""
        state = MagicMock(spec=AgentGraphState)
        state.job_attachments = {}
        state.messages = []
        return state

    @pytest.fixture
    def mock_attachment(self):
        """Create a mock attachment."""
        attachment_id = uuid.uuid4()
        attachment = MagicMock(spec=Attachment)
        attachment.id = attachment_id
        attachment.model_dump = MagicMock(
            return_value={"ID": str(attachment_id), "name": "test.pdf", "size": 1024}
        )
        return attachment

    @pytest.mark.asyncio
    async def test_tool_without_args_schema(
        self, mock_tool, mock_tool_call, mock_state
    ):
        """Test that tool is invoked normally when args_schema is None."""
        mock_tool.args_schema = None

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, mock_tool_call, mock_state)

        assert result == {"result": "success"}
        mock_tool.ainvoke.assert_awaited_once_with(mock_tool_call["args"])

    @pytest.mark.asyncio
    async def test_tool_with_dict_args_schema(
        self, mock_tool, mock_tool_call, mock_state
    ):
        """Test that tool is invoked normally when args_schema is a dict."""
        mock_tool.args_schema = {"type": "object", "properties": {}}

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, mock_tool_call, mock_state)

        assert result == {"result": "success"}
        mock_tool.ainvoke.assert_awaited_once_with(mock_tool_call["args"])

    @pytest.mark.asyncio
    async def test_tool_with_non_basemodel_schema(
        self, mock_tool, mock_tool_call, mock_state
    ):
        """Test that tool is invoked normally when args_schema is not a BaseModel."""
        mock_tool.args_schema = str

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, mock_tool_call, mock_state)

        assert result == {"result": "success"}
        mock_tool.ainvoke.assert_awaited_once_with(mock_tool_call["args"])

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_no_attachment_paths(
        self,
        mock_get_paths,
        mock_tool,
        mock_tool_call,
        mock_state,
    ):
        """Test that tool is invoked normally when no attachment paths are found."""
        mock_tool.args_schema = MockAttachmentSchema
        mock_get_paths.return_value = []

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, mock_tool_call, mock_state)

        assert result == {"result": "success"}
        mock_get_paths.assert_called_once_with(MockAttachmentSchema)
        mock_tool.ainvoke.assert_awaited_once_with(mock_tool_call["args"])

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_valid_attachments(
        self,
        mock_get_paths,
        mock_tool,
        mock_attachment,
        mock_state,
    ):
        """Test that tool is invoked with replaced values when all attachments are valid."""
        mock_tool.args_schema = MockAttachmentSchema
        mock_get_paths.return_value = ["$.attachment"]

        # Setup state with valid attachment (string keys)
        mock_state.job_attachments = {str(mock_attachment.id): mock_attachment}

        # Setup tool call with attachment ID
        tool_call = cast(
            ToolCall,
            {
                "name": "test_tool",
                "args": {"attachment": {"ID": str(mock_attachment.id)}, "name": "test"},
                "id": "call_123",
            },
        )

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, tool_call, mock_state)

        assert result == {"result": "success"}
        # Verify that tool.ainvoke was called (with replaced attachment)
        mock_tool.ainvoke.assert_awaited_once()
        called_args = mock_tool.ainvoke.call_args[0][0]
        assert called_args["name"] == "test"
        assert "attachment" in called_args

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_missing_attachment(
        self,
        mock_get_paths,
        mock_tool,
        mock_state,
    ):
        """Test that error is returned when attachment is missing from state."""
        mock_tool.args_schema = MockAttachmentSchema
        mock_get_paths.return_value = ["$.attachment"]

        attachment_id = uuid.uuid4()
        tool_call = cast(
            ToolCall,
            {
                "name": "test_tool",
                "args": {"attachment": {"ID": str(attachment_id)}, "name": "test"},
                "id": "call_123",
            },
        )

        # Empty state - attachment not found
        mock_state.job_attachments = {}

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, tool_call, mock_state)

        assert isinstance(result, dict)
        assert "error" in result
        assert str(attachment_id) in result["error"]
        assert "Could not find JobAttachment" in result["error"]
        mock_tool.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_multiple_missing_attachments(
        self,
        mock_get_paths,
        mock_tool,
        mock_state,
    ):
        """Test that all missing attachments are reported in error."""
        mock_tool.args_schema = MockAttachmentSchema
        mock_get_paths.return_value = ["$.attachments[*]"]

        attachment_id_1 = uuid.uuid4()
        attachment_id_2 = uuid.uuid4()

        tool_call = cast(
            ToolCall,
            {
                "name": "test_tool",
                "args": {
                    "attachments": [
                        {"ID": str(attachment_id_1)},
                        {"ID": str(attachment_id_2)},
                    ],
                    "name": "test",
                },
                "id": "call_123",
            },
        )

        # Empty state - both attachments not found
        mock_state.job_attachments = {}

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, tool_call, mock_state)

        assert isinstance(result, dict)
        assert "error" in result

        # Check that both attachment IDs are in the error message
        assert str(attachment_id_1) in result["error"]
        assert str(attachment_id_2) in result["error"]

        # Check that errors are newline-separated
        error_lines = result["error"].split("\n")
        assert len(error_lines) == 2

        mock_tool.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_invalid_uuid(
        self,
        mock_get_paths,
        mock_tool,
        mock_state,
    ):
        """Test that error is returned when attachment ID is not a valid UUID."""
        mock_tool.args_schema = MockAttachmentSchema
        mock_get_paths.return_value = ["$.attachment"]

        invalid_id = "not-a-valid-uuid"
        tool_call = cast(
            ToolCall,
            {
                "name": "test_tool",
                "args": {"attachment": {"ID": invalid_id}, "name": "test"},
                "id": "call_123",
            },
        )

        mock_state.job_attachments = {}

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, tool_call, mock_state)

        assert isinstance(result, dict)
        assert "error" in result
        assert invalid_id in result["error"]
        mock_tool.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_partial_valid_attachments(
        self,
        mock_get_paths,
        mock_tool,
        mock_attachment,
        mock_state,
    ):
        """Test that error is returned when some attachments are valid and others are not."""
        mock_tool.args_schema = MockAttachmentSchema
        mock_get_paths.return_value = ["$.attachments[*]"]

        # One valid, one invalid (string keys)
        mock_state.job_attachments = {str(mock_attachment.id): mock_attachment}
        invalid_id = uuid.uuid4()

        tool_call = cast(
            ToolCall,
            {
                "name": "test_tool",
                "args": {
                    "attachments": [
                        {"ID": str(mock_attachment.id)},
                        {"ID": str(invalid_id)},
                    ],
                    "name": "test",
                },
                "id": "call_123",
            },
        )

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, tool_call, mock_state)

        assert isinstance(result, dict)
        assert "error" in result
        assert str(invalid_id) in result["error"]
        assert str(mock_attachment.id) not in result["error"]
        mock_tool.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_complex_nested_structure(
        self,
        mock_get_paths,
        mock_tool,
        mock_state,
    ):
        """Test attachment validation with complex nested object structures and deep paths."""
        mock_tool.args_schema = MockAttachmentSchema

        # Setup multiple attachments with different IDs
        attachment1_id = uuid.uuid4()
        attachment2_id = uuid.uuid4()
        attachment3_id = uuid.uuid4()
        missing_attachment_id = uuid.uuid4()

        attachment1 = MagicMock(spec=Attachment)
        attachment1.id = attachment1_id
        attachment1.model_dump = MagicMock(
            return_value={
                "ID": str(attachment1_id),
                "name": "document1.pdf",
                "size": 1024,
            }
        )

        attachment2 = MagicMock(spec=Attachment)
        attachment2.id = attachment2_id
        attachment2.model_dump = MagicMock(
            return_value={
                "ID": str(attachment2_id),
                "name": "document2.pdf",
                "size": 2048,
            }
        )

        attachment3 = MagicMock(spec=Attachment)
        attachment3.id = attachment3_id
        attachment3.model_dump = MagicMock(
            return_value={
                "ID": str(attachment3_id),
                "name": "document3.pdf",
                "size": 3072,
            }
        )

        # Setup state with available attachments (string keys)
        mock_state.job_attachments = {
            str(attachment1_id): attachment1,
            str(attachment2_id): attachment2,
            str(attachment3_id): attachment3,
        }

        # Define complex nested paths
        mock_get_paths.return_value = [
            "$.request.metadata.primary_attachment",
            "$.request.documents[*]",
            "$.workflow.steps[*].input_files[*]",
            "$.backup.archive.files[*]",
        ]

        # Create complex nested tool call structure
        tool_call = cast(
            ToolCall,
            {
                "name": "complex_tool",
                "args": {
                    "request": {
                        "metadata": {
                            "primary_attachment": {"ID": str(attachment1_id)},
                            "description": "Main request",
                        },
                        "documents": [
                            {"ID": str(attachment2_id)},
                            {"ID": str(missing_attachment_id)},  # This one is missing
                        ],
                    },
                    "workflow": {
                        "name": "process_docs",
                        "steps": [
                            {
                                "name": "step1",
                                "input_files": [{"ID": str(attachment3_id)}],
                            },
                            {
                                "name": "step2",
                                "input_files": [
                                    {"ID": str(attachment1_id)},
                                ],
                            },
                        ],
                    },
                    "backup": {
                        "archive": {
                            "files": [
                                {"ID": str(attachment2_id)},
                            ]
                        }
                    },
                    "other_field": "some_value",
                },
                "id": "call_complex_123",
            },
        )

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, tool_call, mock_state)

        # Should return error for the missing attachment
        assert isinstance(result, dict)
        assert "error" in result
        assert str(missing_attachment_id) in result["error"]
        assert "Could not find JobAttachment" in result["error"]

        # Valid attachments should not be in error message
        assert str(attachment1_id) not in result["error"]
        assert str(attachment2_id) not in result["error"]
        assert str(attachment3_id) not in result["error"]

        # Tool should not be invoked due to error
        mock_tool.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    @patch(
        "uipath_langchain.agent.wrappers.job_attachment_wrapper.get_job_attachment_paths"
    )
    async def test_tool_with_complex_nested_structure_all_valid(
        self,
        mock_get_paths,
        mock_tool,
        mock_state,
    ):
        """Test successful replacement with complex nested structure when all attachments are valid."""
        mock_tool.args_schema = MockAttachmentSchema

        # Setup multiple attachments
        attachment1_id = uuid.uuid4()
        attachment2_id = uuid.uuid4()
        attachment3_id = uuid.uuid4()

        attachment1 = MagicMock(spec=Attachment)
        attachment1.id = attachment1_id
        attachment1.model_dump = MagicMock(
            return_value={
                "ID": str(attachment1_id),
                "name": "document1.pdf",
                "size": 1024,
            }
        )

        attachment2 = MagicMock(spec=Attachment)
        attachment2.id = attachment2_id
        attachment2.model_dump = MagicMock(
            return_value={
                "ID": str(attachment2_id),
                "name": "document2.pdf",
                "size": 2048,
            }
        )

        attachment3 = MagicMock(spec=Attachment)
        attachment3.id = attachment3_id
        attachment3.model_dump = MagicMock(
            return_value={
                "ID": str(attachment3_id),
                "name": "document3.pdf",
                "size": 3072,
            }
        )

        # Setup state with all attachments (string keys)
        mock_state.job_attachments = {
            str(attachment1_id): attachment1,
            str(attachment2_id): attachment2,
            str(attachment3_id): attachment3,
        }

        # Define complex nested paths
        mock_get_paths.return_value = [
            "$.request.metadata.primary_attachment",
            "$.request.documents[*]",
            "$.workflow.steps[*].input_files[*]",
        ]

        # Create complex nested tool call structure with all valid attachments
        tool_call = cast(
            ToolCall,
            {
                "name": "complex_tool",
                "args": {
                    "request": {
                        "metadata": {
                            "primary_attachment": {"ID": str(attachment1_id)},
                            "description": "Main request",
                        },
                        "documents": [
                            {"ID": str(attachment2_id)},
                            {"ID": str(attachment3_id)},
                        ],
                    },
                    "workflow": {
                        "name": "process_docs",
                        "steps": [
                            {
                                "name": "step1",
                                "input_files": [{"ID": str(attachment1_id)}],
                            },
                            {
                                "name": "step2",
                                "input_files": [{"ID": str(attachment2_id)}],
                            },
                        ],
                    },
                    "other_field": "some_value",
                },
                "id": "call_complex_456",
            },
        )

        wrapper = get_job_attachment_wrapper()
        result = await wrapper(mock_tool, tool_call, mock_state)

        # Should succeed without errors
        assert result == {"result": "success"}

        # Tool should be invoked with replaced attachments
        mock_tool.ainvoke.assert_awaited_once()
        called_args = mock_tool.ainvoke.call_args[0][0]

        # Verify structure is preserved
        assert "request" in called_args
        assert "metadata" in called_args["request"]
        assert "documents" in called_args["request"]
        assert "workflow" in called_args
        assert "steps" in called_args["workflow"]
        assert called_args["other_field"] == "some_value"

        # Verify attachments were replaced (they should now be full objects)
        primary_attachment = called_args["request"]["metadata"]["primary_attachment"]
        assert isinstance(primary_attachment, dict)
        assert "name" in primary_attachment or "ID" in primary_attachment
