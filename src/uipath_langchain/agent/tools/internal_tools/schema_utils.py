"""Utility functions for internal tool schema manipulation."""

from typing import Any


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
