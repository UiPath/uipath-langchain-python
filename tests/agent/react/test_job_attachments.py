import uuid
from typing import Any

import pytest
from jsonschema_pydantic_converter import transform_with_modules
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from uipath.platform.attachments import Attachment
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.job_attachments import (
    get_job_attachments,
    parse_attachments_from_conversation_messages,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.reducers import (
    merge_dicts,
)


class TestGetJobAttachments:
    """Test job attachment extraction from data based on schema."""

    def test_base_model_schema(self):
        """Should return empty list when schema is BaseModel (no fields)."""
        data = {"name": "test", "value": 42}

        result = get_job_attachments(BaseModel, data)

        assert result == []

    def test_no_attachments_in_schema(self):
        """Should return empty list when schema has no job-attachment fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
        }
        model = create_model(schema)
        data = {"name": "test", "value": 42}

        result = get_job_attachments(model, data)

        assert result == []

    def test_no_attachments_in_data(self):
        """Should return empty list when data has no attachment values."""
        schema = {
            "type": "object",
            "properties": {"attachment": {"$ref": "#/definitions/job-attachment"}},
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                }
            },
        }
        model = create_model(schema)
        data: dict[str, Any] = {}

        result = get_job_attachments(model, data)

        assert result == []

    def test_single_direct_attachment(self):
        """Should extract single direct attachment field."""
        schema = {
            "type": "object",
            "properties": {"attachment": {"$ref": "#/definitions/job-attachment"}},
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID"],
                }
            },
        }
        model = create_model(schema)
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        data = {
            "attachment": {
                "ID": test_uuid,
                "FullName": "document.pdf",
                "MimeType": "application/pdf",
            }
        }

        result = get_job_attachments(model, data)

        assert len(result) == 1
        assert str(result[0].id) == test_uuid
        assert result[0].full_name == "document.pdf"
        assert result[0].mime_type == "application/pdf"

    def test_multiple_attachments_in_array(self):
        """Should extract all attachments from array field."""
        schema = {
            "type": "object",
            "properties": {
                "attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)
        uuid1 = "550e8400-e29b-41d4-a716-446655440001"
        uuid2 = "550e8400-e29b-41d4-a716-446655440002"
        uuid3 = "550e8400-e29b-41d4-a716-446655440003"
        data = {
            "attachments": [
                {"ID": uuid1, "FullName": "file1.pdf", "MimeType": "application/pdf"},
                {
                    "ID": uuid2,
                    "FullName": "file2.docx",
                    "MimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                },
                {
                    "ID": uuid3,
                    "FullName": "file3.xlsx",
                    "MimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                },
            ]
        }

        result = get_job_attachments(model, data)

        assert len(result) == 3
        assert str(result[0].id) == uuid1
        assert result[0].full_name == "file1.pdf"
        assert str(result[1].id) == uuid2
        assert result[1].full_name == "file2.docx"
        assert str(result[2].id) == uuid3
        assert result[2].full_name == "file3.xlsx"

    def test_mixed_direct_and_array_attachments(self):
        """Should extract attachments from both direct and array fields."""
        schema = {
            "type": "object",
            "properties": {
                "primary_attachment": {"$ref": "#/definitions/job-attachment"},
                "additional_attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                },
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID"],
                }
            },
        }
        model = create_model(schema)
        uuid_primary = "550e8400-e29b-41d4-a716-446655440010"
        uuid1 = "550e8400-e29b-41d4-a716-446655440011"
        uuid2 = "550e8400-e29b-41d4-a716-446655440012"
        data = {
            "primary_attachment": {
                "ID": uuid_primary,
                "FullName": "main.pdf",
                "MimeType": "application/pdf",
            },
            "additional_attachments": [
                {"ID": uuid1, "FullName": "extra1.pdf", "MimeType": "application/pdf"},
                {"ID": uuid2, "FullName": "extra2.pdf", "MimeType": "application/pdf"},
            ],
        }

        result = get_job_attachments(model, data)

        assert len(result) == 3
        # Check that all attachments are extracted (order may vary based on schema field order)
        ids = {str(att.id) for att in result}
        assert ids == {uuid_primary, uuid1, uuid2}

    def test_empty_array_attachments(self):
        """Should handle empty attachment arrays gracefully."""
        schema = {
            "type": "object",
            "properties": {
                "attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)
        data: dict[str, Any] = {"attachments": []}

        result = get_job_attachments(model, data)

        assert result == []

    def test_optional_attachment_field(self):
        """Should handle optional attachment fields that are not present."""
        schema = {
            "type": "object",
            "properties": {
                "attachment": {"$ref": "#/definitions/job-attachment"},
                "other_field": {"type": "string"},
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)
        data = {"other_field": "value"}

        result = get_job_attachments(model, data)

        assert result == []

    def test_pydantic_model_input(self):
        """Should handle Pydantic model instances as input data."""
        schema = {
            "type": "object",
            "properties": {"attachment": {"$ref": "#/definitions/job-attachment"}},
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)

        # Create a Pydantic model instance
        class TestModel(BaseModel):
            attachment: dict[str, Any]

        test_uuid = "550e8400-e29b-41d4-a716-446655440099"
        data_model = TestModel(
            attachment={
                "ID": test_uuid,
                "FullName": "test.pdf",
                "MimeType": "application/pdf",
            }
        )

        result = get_job_attachments(model, data_model)

        assert len(result) == 1
        assert str(result[0].id) == test_uuid
        assert result[0].full_name == "test.pdf"

    def test_attachment_with_additional_fields(self):
        """Should extract attachments with additional optional fields."""
        schema = {
            "type": "object",
            "properties": {"attachment": {"$ref": "#/definitions/job-attachment"}},
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                        "size": {"type": "integer"},
                    },
                    "required": ["FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)
        test_uuid = "550e8400-e29b-41d4-a716-446655440100"
        data = {
            "attachment": {
                "ID": test_uuid,
                "FullName": "document.pdf",
                "MimeType": "application/pdf",
                "size": 1024,
            }
        }

        result = get_job_attachments(model, data)

        assert len(result) == 1
        assert str(result[0].id) == test_uuid
        assert result[0].full_name == "document.pdf"
        assert result[0].mime_type == "application/pdf"

    def test_nested_structure_with_attachments(self):
        """Should extract attachments from nested structures."""
        schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {
                        "attachment": {"$ref": "#/definitions/job-attachment"}
                    },
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID"],
                }
            },
        }
        model, _ = transform_with_modules(schema)
        test_uuid = "550e8400-e29b-41d4-a716-446655440200"
        data = {
            "result": {
                "attachment": {
                    "ID": test_uuid,
                    "FullName": "nested.pdf",
                    "MimeType": "application/pdf",
                }
            }
        }

        result = get_job_attachments(model, data)

        # Implementation now traverses nested objects
        assert len(result) == 1
        assert str(result[0].id) == test_uuid
        assert result[0].full_name == "nested.pdf"
        assert result[0].mime_type == "application/pdf"

    def test_deeply_nested_and_array_structures(self):
        """Should extract attachments from deeply nested structures and arrays of nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "files": {
                                        "type": "array",
                                        "items": {
                                            "$ref": "#/definitions/job-attachment"
                                        },
                                    }
                                },
                            },
                        }
                    },
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID"],
                }
            },
        }
        model = create_model(schema)
        uuid1 = "550e8400-e29b-41d4-a716-446655440301"
        uuid2 = "550e8400-e29b-41d4-a716-446655440302"
        uuid3 = "550e8400-e29b-41d4-a716-446655440303"
        data = {
            "data": {
                "items": [
                    {
                        "files": [
                            {
                                "ID": uuid1,
                                "FullName": "file1.pdf",
                                "MimeType": "application/pdf",
                            },
                            {
                                "ID": uuid2,
                                "FullName": "file2.pdf",
                                "MimeType": "application/pdf",
                            },
                        ]
                    },
                    {
                        "files": [
                            {
                                "ID": uuid3,
                                "FullName": "file3.pdf",
                                "MimeType": "application/pdf",
                            }
                        ]
                    },
                ]
            }
        }

        result = get_job_attachments(model, data)

        # Should extract all attachments from deeply nested arrays
        assert len(result) == 3
        ids = {str(att.id) for att in result}
        assert ids == {uuid1, uuid2, uuid3}

    def test_raises_system_error_on_non_uuid_attachment_id(self):
        """A tool-output attachment with a non-UUID ID must fail loud as SYSTEM."""
        schema = {
            "type": "object",
            "properties": {
                "attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID"],
                }
            },
        }
        model = create_model(schema)
        data = {
            "attachments": [
                {"ID": "att_x", "FullName": "bad.pdf", "MimeType": "application/pdf"},
            ]
        }

        with pytest.raises(AgentRuntimeError) as exc_info:
            get_job_attachments(model, data)

        error_info = exc_info.value.error_info
        assert error_info.category == UiPathErrorCategory.SYSTEM
        assert error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.INVALID_ATTACHMENT_ID
        )
        assert "att_x" in error_info.detail

    def test_accepts_attachment_model_dump_output(self):
        """Attachment.model_dump() output should use Attachment validation rules."""
        schema = {
            "type": "object",
            "properties": {
                "attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID", "FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)
        valid_uuid = "550e8400-e29b-41d4-a716-446655440401"
        attachment = Attachment.model_validate(
            {
                "ID": valid_uuid,
                "FullName": "model-dump.pdf",
                "MimeType": "application/pdf",
            }
        )
        data = {"attachments": [attachment.model_dump()]}

        result = get_job_attachments(model, data)

        assert len(result) == 1
        assert str(result[0].id) == valid_uuid
        assert result[0].full_name == "model-dump.pdf"

    def test_accepts_attachment_field_as_nested_model_instance(self):
        """Regression: at runtime tool args are coerced into the generated input
        model, so an attachment arrives as a model instance whose ``Metadata``
        object is a nested sub-model (``DynamicType_*``), not a dict. This must
        not raise (previously failed with "Input should be a valid dictionary").
        """
        schema = {
            "type": "object",
            "properties": {"newArgument": {"$ref": "#/definitions/job-attachment"}},
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "x-uipath-resource-kind": "JobAttachment",
                    "required": ["ID"],
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                        "Metadata": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                }
            },
        }
        model = create_model(schema)
        valid_uuid = "550e8400-e29b-41d4-a716-446655440500"

        # Coerce raw input through the generated model exactly as the runtime
        # does, then pass it inside kwargs (a dict holding a model instance) as
        # process_tool_fn does. Metadata is now a nested model, not a dict.
        validated: Any = model.model_validate(
            {
                "newArgument": {
                    "ID": valid_uuid,
                    "FullName": "workbook.xlsx",
                    "MimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "Metadata": {"size": "38472"},
                }
            }
        )
        assert isinstance(validated.newArgument.Metadata, BaseModel)
        kwargs = {"newArgument": validated.newArgument}

        result = get_job_attachments(model, kwargs)

        assert len(result) == 1
        assert str(result[0].id) == valid_uuid
        assert result[0].metadata == {"size": "38472"}

    def test_raises_system_error_with_field_on_invalid_attachment_shape(self):
        """A valid-UUID attachment missing a required field fails loud as SYSTEM,
        naming the offending field in a human-readable message."""
        schema = {
            "type": "object",
            "properties": {
                "attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID"],
                }
            },
        }
        model = create_model(schema)
        valid_uuid = "550e8400-e29b-41d4-a716-446655440400"
        data = {
            "attachments": [
                {"ID": valid_uuid, "FullName": "doc.pdf"},  # MimeType missing
            ]
        }

        with pytest.raises(AgentRuntimeError) as exc_info:
            get_job_attachments(model, data)

        error_info = exc_info.value.error_info
        assert error_info.category == UiPathErrorCategory.SYSTEM
        assert error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.OUTPUT_VALIDATION_ERROR
        )
        assert "MimeType" in error_info.detail or "mime_type" in error_info.detail

    def test_raises_output_validation_error_when_attachment_id_is_missing(self):
        """A missing attachment id is an invalid shape, not an invalid UUID."""
        schema = {
            "type": "object",
            "properties": {
                "attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["ID", "FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)
        data = {
            "attachments": [
                {"FullName": "missing-id.pdf", "MimeType": "application/pdf"},
            ]
        }

        with pytest.raises(AgentRuntimeError) as exc_info:
            get_job_attachments(model, data)

        error_info = exc_info.value.error_info
        assert error_info.category == UiPathErrorCategory.SYSTEM
        assert error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.OUTPUT_VALIDATION_ERROR
        )
        assert "ID" in error_info.detail or "id" in error_info.detail

    def test_filters_out_none_attachments_in_array(self):
        """Should filter out None items from attachment arrays."""
        schema = {
            "type": "object",
            "properties": {
                "attachments": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/job-attachment"},
                }
            },
            "definitions": {
                "job-attachment": {
                    "type": "object",
                    "properties": {
                        "ID": {"type": "string"},
                        "FullName": {"type": "string"},
                        "MimeType": {"type": "string"},
                    },
                    "required": ["FullName", "MimeType"],
                }
            },
        }
        model = create_model(schema)
        uuid1 = "550e8400-e29b-41d4-a716-446655440001"
        uuid2 = "550e8400-e29b-41d4-a716-446655440002"
        data = {
            "attachments": [
                {"ID": uuid1, "FullName": "file1.pdf", "MimeType": "application/pdf"},
                None,  # This should be filtered out
                {
                    "ID": uuid2,
                    "FullName": "file2.docx",
                    "MimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                },
                None,  # This should also be filtered out
            ]
        }

        result = get_job_attachments(model, data)

        # Should only get 2 attachments, None items should be filtered out
        assert len(result) == 2
        assert str(result[0].id) == uuid1
        assert result[0].full_name == "file1.pdf"
        assert str(result[1].id) == uuid2
        assert result[1].full_name == "file2.docx"


class TestMergeDicts:
    """Test dictionary merging."""

    def test_both_empty_dictionaries(self):
        """Should return empty dict when both inputs are empty."""
        left: dict[str, Attachment] = {}
        right: dict[str, Attachment] = {}

        result = merge_dicts(left, right)

        assert result == {}

    def test_left_empty_right_has_attachments(self):
        """Should return right dict when left is empty."""
        uuid1 = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        right = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "file1.pdf",
                    "MimeType": "application/pdf",
                }
            )
        }

        result = merge_dicts({}, right)

        assert result == right
        assert len(result) == 1
        assert result[str(uuid1)].full_name == "file1.pdf"

    def test_left_has_attachments_right_empty(self):
        """Should return left dict when right is empty."""
        uuid1 = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        left = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "file1.pdf",
                    "MimeType": "application/pdf",
                }
            )
        }

        result = merge_dicts(left, {})

        assert result == left
        assert len(result) == 1
        assert result[str(uuid1)].full_name == "file1.pdf"

    def test_no_overlapping_uuids(self):
        """Should merge dicts with no overlapping keys."""
        uuid1 = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        uuid2 = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")

        left = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "file1.pdf",
                    "MimeType": "application/pdf",
                }
            )
        }
        right = {
            str(uuid2): Attachment.model_validate(
                {
                    "ID": str(uuid2),
                    "FullName": "file2.pdf",
                    "MimeType": "application/pdf",
                }
            )
        }

        result = merge_dicts(left, right)

        assert len(result) == 2
        assert str(uuid1) in result
        assert str(uuid2) in result
        assert result[str(uuid1)].full_name == "file1.pdf"
        assert result[str(uuid2)].full_name == "file2.pdf"

    def test_overlapping_uuid_right_takes_precedence(self):
        """Should use right value when same UUID exists in both dicts."""
        uuid1 = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")

        left = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "old_file.pdf",
                    "MimeType": "application/pdf",
                }
            )
        }
        right = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "new_file.pdf",
                    "MimeType": "application/pdf",
                }
            )
        }

        result = merge_dicts(left, right)

        assert len(result) == 1
        assert result[str(uuid1)].full_name == "new_file.pdf"  # Right takes precedence

    def test_mixed_overlapping_and_unique(self):
        """Should correctly merge dicts with both overlapping and unique keys."""
        uuid1 = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        uuid2 = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        uuid3 = uuid.UUID("550e8400-e29b-41d4-a716-446655440003")

        left = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "file1_old.pdf",
                    "MimeType": "application/pdf",
                }
            ),
            str(uuid2): Attachment.model_validate(
                {
                    "ID": str(uuid2),
                    "FullName": "file2.pdf",
                    "MimeType": "application/pdf",
                }
            ),
        }
        right = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "file1_new.pdf",
                    "MimeType": "application/pdf",
                }
            ),
            str(uuid3): Attachment.model_validate(
                {
                    "ID": str(uuid3),
                    "FullName": "file3.pdf",
                    "MimeType": "application/pdf",
                }
            ),
        }

        result = merge_dicts(left, right)

        assert len(result) == 3
        assert result[str(uuid1)].full_name == "file1_new.pdf"  # Right overrides
        assert result[str(uuid2)].full_name == "file2.pdf"  # From left only
        assert result[str(uuid3)].full_name == "file3.pdf"  # From right only

    def test_multiple_attachments_same_operation(self):
        """Should handle merging multiple attachments at once."""
        uuid1 = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        uuid2 = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        uuid3 = uuid.UUID("550e8400-e29b-41d4-a716-446655440003")
        uuid4 = uuid.UUID("550e8400-e29b-41d4-a716-446655440004")

        left = {
            str(uuid1): Attachment.model_validate(
                {
                    "ID": str(uuid1),
                    "FullName": "file1.pdf",
                    "MimeType": "application/pdf",
                }
            ),
            str(uuid2): Attachment.model_validate(
                {
                    "ID": str(uuid2),
                    "FullName": "file2.pdf",
                    "MimeType": "application/pdf",
                }
            ),
        }
        right = {
            str(uuid3): Attachment.model_validate(
                {
                    "ID": str(uuid3),
                    "FullName": "file3.pdf",
                    "MimeType": "application/pdf",
                }
            ),
            str(uuid4): Attachment.model_validate(
                {
                    "ID": str(uuid4),
                    "FullName": "file4.pdf",
                    "MimeType": "application/pdf",
                }
            ),
        }

        result = merge_dicts(left, right)

        assert len(result) == 4
        assert all(str(uid) in result for uid in [uuid1, uuid2, uuid3, uuid4])


class TestParseAttachmentsFromConversationMessages:
    """Test parsing attachments from conversation message metadata."""

    def test_empty_messages(self):
        """Should return empty dict for empty messages list."""
        result = parse_attachments_from_conversation_messages([])

        assert result == {}

    def test_no_human_messages(self):
        """Should return empty dict when no HumanMessages present."""
        messages = [
            SystemMessage(content="System prompt"),
            AIMessage(content="AI response"),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert result == {}

    def test_human_message_without_metadata(self):
        """Should skip HumanMessages without attachment metadata."""
        messages = [
            HumanMessage(content="Hello"),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert result == {}

    def test_human_message_with_attachments_in_metadata(self):
        """Should extract attachments from HumanMessage metadata attachments list."""
        attachment_id = "a940a416-b97b-4146-3089-08de5f4d0a87"
        messages = [
            HumanMessage(
                content="Check this file",
                additional_kwargs={
                    "attachments": [
                        {
                            "id": attachment_id,
                            "full_name": "document.pdf",
                            "mime_type": "application/pdf",
                        }
                    ],
                },
            ),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert len(result) == 1
        assert attachment_id in result
        assert result[attachment_id].full_name == "document.pdf"
        assert str(result[attachment_id].id) == attachment_id
        assert result[attachment_id].mime_type == "application/pdf"

    def test_human_message_without_full_name_skipped(self):
        """Should skip attachments without full_name."""
        messages = [
            HumanMessage(
                content="Check this",
                additional_kwargs={
                    "attachments": [
                        {
                            "id": "a940a416-b97b-4146-3089-08de5f4d0a87",
                            "mime_type": "application/pdf",
                        }
                    ],
                },
            ),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert result == {}

    def test_human_message_without_attachment_id_skipped(self):
        """Should skip attachments without id."""
        messages = [
            HumanMessage(
                content="Check this",
                additional_kwargs={
                    "attachments": [
                        {
                            "full_name": "file.pdf",
                            "mime_type": "application/pdf",
                        }
                    ],
                },
            ),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert result == {}

    def test_multiple_attachments_from_multiple_messages(self):
        """Should extract attachments from multiple HumanMessages."""
        id1 = "a940a416-b97b-4146-3089-08de5f4d0a87"
        id2 = "b940b416-c97c-5146-4089-09de6f5d1a88"
        messages = [
            SystemMessage(content="System"),
            HumanMessage(
                content="First file",
                additional_kwargs={
                    "attachments": [
                        {
                            "id": id1,
                            "full_name": "file1.jpg",
                            "mime_type": "image/jpeg",
                        }
                    ],
                },
            ),
            AIMessage(content="Got it"),
            HumanMessage(
                content="Second file",
                additional_kwargs={
                    "attachments": [
                        {
                            "id": id2,
                            "full_name": "file2.pdf",
                            "mime_type": "application/pdf",
                        }
                    ],
                },
            ),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert len(result) == 2
        assert id1 in result
        assert id2 in result
        assert result[id1].full_name == "file1.jpg"
        assert result[id2].full_name == "file2.pdf"

    def test_multiple_attachments_in_single_message(self):
        """Should extract multiple attachments from a single HumanMessage."""
        id1 = "a940a416-b97b-4146-3089-08de5f4d0a87"
        id2 = "b940b416-c97c-5146-4089-09de6f5d1a88"
        messages = [
            HumanMessage(
                content="Check these files",
                additional_kwargs={
                    "attachments": [
                        {
                            "id": id1,
                            "full_name": "file1.jpg",
                            "mime_type": "image/jpeg",
                        },
                        {
                            "id": id2,
                            "full_name": "file2.pdf",
                            "mime_type": "application/pdf",
                        },
                    ],
                },
            ),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert len(result) == 2
        assert result[id1].full_name == "file1.jpg"
        assert result[id2].full_name == "file2.pdf"

    def test_empty_attachments_list(self):
        """Should return empty dict when attachments list is empty."""
        messages = [
            HumanMessage(
                content="Hello",
                additional_kwargs={"attachments": []},
            ),
        ]

        result = parse_attachments_from_conversation_messages(messages)

        assert result == {}
