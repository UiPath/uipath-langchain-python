import uuid
from typing import Any

from pydantic import BaseModel
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.job_attachments import get_job_attachments
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.utils import (
    add_job_attachments,
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
        model = create_model(schema)
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


class TestAddJobAttachments:
    """Test attachment dictionary merging."""

    def test_both_empty_dictionaries(self):
        """Should return empty dict when both inputs are empty."""
        left: dict[str, Attachment] = {}
        right: dict[str, Attachment] = {}

        result = add_job_attachments(left, right)

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

        result = add_job_attachments({}, right)

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

        result = add_job_attachments(left, {})

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

        result = add_job_attachments(left, right)

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

        result = add_job_attachments(left, right)

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

        result = add_job_attachments(left, right)

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

        result = add_job_attachments(left, right)

        assert len(result) == 4
        assert all(str(uid) in result for uid in [uuid1, uuid2, uuid3, uuid4])
