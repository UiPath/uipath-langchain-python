"""Standalone ontology context tool.

Grouped in its own subpackage so it does not touch the entity tool's
``datafabric_tool``/``datafabric_subgraph``. The agent definition names one
ontology per context node; this tool grounds an inner SQL sub-graph on the
ontology (its schema files or its graph, per :data:`ONTOLOGY_SOURCE_FF`) and runs
the model's SQL on the Ontology service. The low-code context is feature-flag
gated by :data:`DATAFABRIC_ONTOLOGY_FF`; a coded agent builds the same tool with
:func:`create_ontology_tool`.
"""

from .ontology_grounding import ONTOLOGY_SOURCE_FF, OntologySource
from .ontology_tool import (
    DATAFABRIC_ONTOLOGY_FF,
    create_datafabric_ontology_tool,
    create_ontology_tool,
)

__all__ = [
    "DATAFABRIC_ONTOLOGY_FF",
    "ONTOLOGY_SOURCE_FF",
    "OntologySource",
    "create_datafabric_ontology_tool",
    "create_ontology_tool",
]
