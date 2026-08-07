"""Compatibility exports for agent JSON utilities."""

from ..json_utils import (
    _coerce_field,
    _create_type_matcher,
    _get_target_type,
    _is_pydantic_model,
    _json_key,
    _unwrap_lists,
    _unwrap_optional,
    coerce_json_strings,
    extract_values_by_paths,
    get_json_paths_by_type,
)

__all__ = [
    "_coerce_field",
    "_create_type_matcher",
    "_get_target_type",
    "_is_pydantic_model",
    "_json_key",
    "_unwrap_lists",
    "_unwrap_optional",
    "coerce_json_strings",
    "extract_values_by_paths",
    "get_json_paths_by_type",
]
