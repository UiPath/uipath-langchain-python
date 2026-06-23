"""Data Fabric tool module for entity-based SQL queries."""

from .datafabric_tool import (
    create_datafabric_query_tool,
    resolve_context_ontologies,
)

__all__ = [
    "create_datafabric_query_tool",
    "resolve_context_ontologies",
]
