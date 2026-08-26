"""Backward-compatible imports for agent JSON utilities."""

from ..attachments.pydantic_json import (
    coerce_json_strings,
    extract_values_by_paths,
    get_json_paths_by_type,
)

__all__ = [
    "coerce_json_strings",
    "extract_values_by_paths",
    "get_json_paths_by_type",
]
