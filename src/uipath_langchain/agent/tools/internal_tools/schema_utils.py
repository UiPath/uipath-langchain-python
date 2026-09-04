"""Utility functions for internal tool schema manipulation."""

from typing import Any

# The `job-attachment` definitions key is load-bearing: the JSON-schema-to-Pydantic
# converter derives the `__Job_attachment` marker type from it, and that marker is
# what `get_job_attachment_paths` looks for when discovering attachment fields.
JOB_ATTACHMENT_DEFINITION: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ID": {"type": "string", "description": "Orchestrator attachment key"},
        "FullName": {"type": "string", "description": "File name"},
        "MimeType": {
            "type": "string",
            "description": "The MIME type of the content",
        },
        "Metadata": {
            "type": "object",
            "description": "Dictionary<string, string> of metadata",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["ID", "FullName", "MimeType"],
    "x-uipath-resource-kind": "JobAttachment",
}

# BatchTransform output schema with file attachment
BATCH_TRANSFORM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "result": {
            "$ref": "#/definitions/job-attachment",
            "description": "The transformed result file as an attachment",
        }
    },
    "required": ["result"],
    "definitions": {"job-attachment": JOB_ATTACHMENT_DEFINITION},
}


def add_query_field_to_schema(
    input_schema: dict[str, Any],
    query_description: str | None = None,
    default_description: str = "Query or prompt for the operation.",
) -> None:
    """Add a dynamic query field to an input schema.

    This modifies the input schema in-place by adding a 'query' property
    and marking it as required.

    Args:
        input_schema: The JSON schema dict to modify
        query_description: Custom description for the query field
        default_description: Default description if query_description is not provided
    """
    if "properties" not in input_schema:
        input_schema["properties"] = {}

    input_schema["properties"]["query"] = {
        "type": "string",
        "description": query_description if query_description else default_description,
    }

    if "required" not in input_schema:
        input_schema["required"] = []

    if "query" not in input_schema["required"]:
        input_schema["required"].append("query")
