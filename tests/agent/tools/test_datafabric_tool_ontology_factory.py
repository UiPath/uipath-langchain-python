"""Tests for the ontology_set → (name, folder) mapping in the DF tool factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from uipath.platform.entities import DataFabricEntityItem

from uipath_langchain.agent.tools.datafabric_tool.datafabric_tool import (
    create_datafabric_query_tool,
)


def _resource(ontology_set):
    entity = DataFabricEntityItem.model_validate(
        {"id": "e1", "referenceKey": "e1", "name": "LibraryLoan", "folderId": "f1"}
    )
    return SimpleNamespace(
        entity_set=[entity],
        ontology_set=ontology_set,
        description="ctx",
    )


def test_factory_maps_ontology_set_and_derives_folder():
    # ontology with no folderId inherits the single entity folder.
    resource = _resource([SimpleNamespace(name="library", folder_key=None)])

    tool = create_datafabric_query_tool(resource, MagicMock())  # type: ignore[arg-type]

    assert tool.coroutine._ontologies == [("library", "f1")]


def test_factory_keeps_per_ontology_folder():
    resource = _resource([SimpleNamespace(name="finance", folder_key="f2")])

    tool = create_datafabric_query_tool(resource, MagicMock())  # type: ignore[arg-type]

    assert tool.coroutine._ontologies == [("finance", "f2")]


def test_factory_no_ontologies_is_empty():
    resource = _resource([])

    tool = create_datafabric_query_tool(resource, MagicMock())  # type: ignore[arg-type]

    assert tool.coroutine._ontologies == []
