"""Versioned SQL prompt templates for Data Fabric entity queries.

Each version is a (SqlPromptContext model, template string) pair:
- The Pydantic model defines the variables the template expects, with defaults.
- The template is a Python format-string rendered via ``template.format_map(ctx.model_dump())``.

SQL_CONSTRAINTS is NOT templated — it is appended verbatim by the prompt builder.
"""

from .context import SqlPromptContext
from .registry import (
    PromptVersion,
    build_prompt_context,
    get_prompt_version,
    list_prompt_versions,
)

__all__ = [
    "PromptVersion",
    "SqlPromptContext",
    "build_prompt_context",
    "get_prompt_version",
    "list_prompt_versions",
]
