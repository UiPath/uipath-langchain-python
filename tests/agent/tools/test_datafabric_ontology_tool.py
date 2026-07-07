"""Tests for the standalone ontology tool (datafabric_tool/datafabric_ontology_tool.py).

Covers the factory (reads ontology_set → handler) and the b2 resolver
(folderPath→key caching, name→Entity, folders_map construction).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from uipath.agent.models.agent import AgentContextResourceConfig

from uipath_langchain.agent.tools.datafabric_tool import datafabric_ontology_tool
from uipath_langchain.agent.tools.datafabric_tool.datafabric_ontology_tool import (
    DataFabricOntologyQueryHandler,
    create_datafabric_ontology_tool,
    resolve_ontology_entities,
)

# --- factory ----------------------------------------------------------------


def _ontology_resource(ontology_set):
    return AgentContextResourceConfig.model_validate(
        {
            "$resourceType": "context",
            "name": "California Schools Ontology",
            "description": "schools domain",
            "contextType": "datafabricontology",
            "ontologySet": ontology_set,
        }
    )


def test_factory_builds_handler_from_ontology_set():
    resource = _ontology_resource(
        [
            {"name": "california_schools", "folderId": "f1"},
            {"name": "finance", "folderId": "f2"},
        ]
    )
    tool = create_datafabric_ontology_tool(resource, MagicMock())

    assert tool.name == "query_datafabric_ontology"
    assert tool.coroutine._ontologies == [  # type: ignore[attr-defined]
        ("california_schools", "f1"),
        ("finance", "f2"),
    ]
    assert "california_schools" in tool.description
    assert "finance" in tool.description


def test_factory_empty_ontology_set():
    tool = create_datafabric_ontology_tool(_ontology_resource([]), MagicMock())
    assert tool.coroutine._ontologies == []  # type: ignore[attr-defined]


# --- resolver (b2) ----------------------------------------------------------


async def test_resolve_builds_folders_map_and_caches_folder_key(monkeypatch):
    folder_keys = {"F/a": "key-a", "F/b": "key-b"}

    sdk = MagicMock()
    sdk.folders.retrieve_key_async = AsyncMock(
        side_effect=lambda folder_path: folder_keys[folder_path]
    )
    sdk.entities.retrieve_by_name_async = AsyncMock(
        side_effect=lambda name, folder_key: SimpleNamespace(name=name)
    )

    captured: dict[str, object] = {}
    fake_service = object()

    def fake_entities_service(**kwargs):
        captured.update(kwargs)
        return fake_service

    monkeypatch.setattr(
        datafabric_ontology_tool, "EntitiesService", fake_entities_service
    )

    entities, service = await resolve_ontology_entities(
        sdk, [("alpha", "F/a"), ("beta", "F/b"), ("gamma", "F/a")]
    )

    assert [e.name for e in entities] == ["alpha", "beta", "gamma"]
    assert captured["folders_map"] == {
        "alpha": "key-a",
        "beta": "key-b",
        "gamma": "key-a",
    }
    assert service is fake_service
    # F/a resolved once (cached), F/b once → 2 folder lookups for 3 entities.
    assert sdk.folders.retrieve_key_async.await_count == 2
    assert sdk.entities.retrieve_by_name_async.await_count == 3


async def test_ensure_graph_guarded_when_flag_off(monkeypatch):
    # Defense in depth: even if the tool is somehow constructed, the handler
    # refuses to do any ontology work when the flag is disabled.
    from uipath.core.feature_flags import FeatureFlags

    monkeypatch.setattr(FeatureFlags, "is_flag_enabled", lambda *a, **k: False)
    handler = DataFabricOntologyQueryHandler(
        ontologies=[("california-schools", "f1")], llm=MagicMock()
    )
    with pytest.raises(ValueError, match="disabled"):
        await handler._ensure_graph()


async def test_resolve_raises_on_unresolved_folder(monkeypatch):
    sdk = MagicMock()
    sdk.folders.retrieve_key_async = AsyncMock(return_value=None)
    monkeypatch.setattr(datafabric_ontology_tool, "EntitiesService", MagicMock())

    try:
        await resolve_ontology_entities(sdk, [("alpha", "F/missing")])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "could not be resolved" in str(e)
