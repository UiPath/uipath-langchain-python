"""Utility functions for internal tool schema manipulation."""

from typing import Any

JOB_ATTACHMENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"outputAttachment": {"$ref": "#/definitions/job-attachment"}},
    "definitions": {
        "job-attachment": {
            "type": "object",
            "required": ["ID"],
            "x-uipath-resource-kind": "JobAttachment",
            "properties": {
                "ID": {
                    "type": "string",
                    "description": "Orchestrator attachment key",
                },
                "FullName": {
                    "type": "string",
                    "description": "File name",
                },
                "MimeType": {
                    "type": "string",
                    "description": 'The MIME type of the content, such as "application/json" or "image/png"',
                },
                "Metadata": {
                    "type": "object",
                    "description": "Dictionary<string, string> of metadata",
                    "additionalProperties": {"type": "string"},
                },
            },
        }
    },
}


def override_array_output_schema(output_schema: dict[str, Any]) -> dict[str, Any]:
    """Override array-type output schemas with job attachment schema.

    Existing agents may have an incorrect output schema configured with
    type "array". This override replaces it with the correct job attachment
    schema.

    Args:
        output_schema: The original output schema dict

    Returns:
        Job attachment schema if original type is "array", otherwise the original schema
    """
    if output_schema.get("type") == "array":
        return dict(JOB_ATTACHMENT_OUTPUT_SCHEMA)
    return output_schema


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
