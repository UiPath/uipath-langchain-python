"""Data Fabric tool module for entity-based SQL queries."""

from .datafabric_tool import (
    DATAFABRIC_ONTOLOGY_FF,
    create_datafabric_query_tool,
    resolve_context_ontologies,
)

__all__ = [
    "DATAFABRIC_ONTOLOGY_FF",
    "create_datafabric_query_tool",
    "resolve_context_ontologies",
]
