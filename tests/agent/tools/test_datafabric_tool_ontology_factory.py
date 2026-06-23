"""Tests for ontology resolution + (name, folder) mapping in the DF tool factory.

Ontologies are standalone ``AgentOntologyResourceConfig`` resources; a Data
Fabric context references them by name via ``ontology_refs``. The caller
resolves those refs to ``(name, folder_key)`` pairs and passes them to the
factory.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentOntologyResourceConfig,
)
from uipath.platform.entities import DataFabricEntityItem

from uipath_langchain.agent.tools.datafabric_tool.datafabric_tool import (
    create_datafabric_query_tool,
    resolve_context_ontologies,
)


def _entity_resource():
    entity = DataFabricEntityItem.model_validate(
        {"id": "e1", "referenceKey": "e1", "name": "LibraryLoan", "folderId": "f1"}
    )
    return SimpleNamespace(entity_set=[entity], description="ctx")


# --- factory: passes resolved ontologies straight through to the handler ---


def test_factory_passes_ontologies_through():
    tool = create_datafabric_query_tool(
        _entity_resource(),  # type: ignore[arg-type]
        MagicMock(),
        ontologies=[("library", "f1")],
    )
    assert tool.coroutine._ontologies == [("library", "f1")]


def test_factory_no_ontologies_is_empty():
    tool = create_datafabric_query_tool(_entity_resource(), MagicMock())  # type: ignore[arg-type]
    assert tool.coroutine._ontologies == []


# --- resolver: ontology_refs → standalone ontology resources → (name, folder) ---


def _ctx(ontology_refs):
    return AgentContextResourceConfig.model_validate(
        {
            "$resourceType": "context",
            "name": "TestDF",
            "description": "",
            "contextType": "datafabricentityset",
            "ontologyRefs": ontology_refs,
        }
    )


def _onto(name, folder_id):
    return AgentOntologyResourceConfig.model_validate(
        {
            "$resourceType": "ontology",
            "name": name,
            "description": "",
            "folderId": folder_id,
        }
    )


def test_resolve_refs_to_name_and_folder():
    ctx = _ctx(["library", "finance"])
    resources = [ctx, _onto("library", "f1"), _onto("finance", "f2")]
    assert resolve_context_ontologies(ctx, resources) == [
        ("library", "f1"),
        ("finance", "f2"),
    ]


def test_resolve_skips_dangling_ref():
    ctx = _ctx(["library", "missing"])
    resources = [ctx, _onto("library", "f1")]
    # 'missing' has no matching ontology resource → skipped, not an error.
    assert resolve_context_ontologies(ctx, resources) == [("library", "f1")]


def test_resolve_no_refs_is_empty():
    ctx = _ctx(None)
    assert resolve_context_ontologies(ctx, [ctx]) == []
