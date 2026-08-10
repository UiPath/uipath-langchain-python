"""Standalone Data Fabric **ontology** tool.

Grouped in its own subpackage so it does not touch the entity tool's
``datafabric_tool``/``datafabric_subgraph``. The agent selects ontologies; this
tool derives its entity allow-list from each ontology's R2RML mapping, grounds an
inner SQL sub-graph on the OWL + R2RML, and reuses the entity tool's execute-SQL
path. Feature-flag gated by :data:`DATAFABRIC_ONTOLOGY_FF`.
"""

from .ontology_tool import DATAFABRIC_ONTOLOGY_FF, create_datafabric_ontology_tool

__all__ = [
    "DATAFABRIC_ONTOLOGY_FF",
    "create_datafabric_ontology_tool",
]
