"""Tests for ontology resolution + (name, folder) mapping in the DF tool factory.

Ontologies are configured inline on the Data Fabric context as a nested
``ontologySet`` (alongside the entity set). The caller resolves those items to
``(name, folder_key)`` pairs and passes them to the factory.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from uipath.agent.models.agent import AgentContextResourceConfig
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
        _entity_resource(),
        MagicMock(),
        ontologies=[("library", "f1")],
    )
    assert tool.coroutine._ontologies == [("library", "f1")]  # type: ignore[attr-defined]


def test_factory_no_ontologies_is_empty():
    tool = create_datafabric_query_tool(_entity_resource(), MagicMock())
    assert tool.coroutine._ontologies == []  # type: ignore[attr-defined]


# --- resolver: nested ontologySet → (name, folder) pairs ---


def _ctx(ontology_set):
    config = {
        "$resourceType": "context",
        "name": "TestDF",
        "description": "",
        "contextType": "datafabricentityset",
    }
    if ontology_set is not None:
        config["ontologySet"] = ontology_set
    return AgentContextResourceConfig.model_validate(config)


def test_resolve_ontology_set_to_name_and_folder():
    ctx = _ctx(
        [
            {"name": "library", "folderId": "f1"},
            {"name": "finance", "folderId": "f2", "referenceKey": "ont-2"},
        ]
    )
    assert resolve_context_ontologies(ctx) == [
        ("library", "f1"),
        ("finance", "f2"),
    ]


def test_resolve_no_ontology_set_is_empty():
    ctx = _ctx(None)
    assert resolve_context_ontologies(ctx) == []
