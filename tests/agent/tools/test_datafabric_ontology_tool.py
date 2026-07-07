"""Tests for the standalone ontology tool (datafabric_tool/ontology/ontology_tool.py).

Covers the factory (reads ontology_set → handler) and the b2 resolver
(folderPath→key caching, name→Entity, folders_map construction).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import uipath.platform as _uipath_platform
from langchain_core.messages import AIMessage, ToolMessage
from uipath.agent.models.agent import AgentContextResourceConfig
from uipath.core.feature_flags import FeatureFlags

from uipath_langchain.agent.tools.datafabric_tool.ontology import (
    ontology_prompt_builder as _opb,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology import (
    ontology_subgraph as _sg,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology import (
    ontology_tool as datafabric_ontology_tool,
)
from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_tool import (
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


# --- _ensure_graph (fetch -> parse -> resolve -> compile) -------------------


def _patch_ensure_graph_deps(monkeypatch, *, fetch=None, parse=None, resolve=None):
    """Patch the flag, SDK, and the fetch/parse/resolve/build/compile seam so
    _ensure_graph can run without any real IO. Returns the sentinel graph."""
    sentinel_graph = object()
    monkeypatch.setattr(FeatureFlags, "is_flag_enabled", lambda *a, **k: True)
    monkeypatch.setattr(_uipath_platform, "UiPath", lambda: MagicMock())
    monkeypatch.setattr(
        datafabric_ontology_tool,
        "fetch_ontology_file",
        fetch or AsyncMock(return_value=("BODY", "text/turtle")),
    )
    monkeypatch.setattr(
        datafabric_ontology_tool,
        "parse_r2rml_entities",
        parse or (lambda _txt: [("frpm", "F/a")]),
    )
    monkeypatch.setattr(
        datafabric_ontology_tool,
        "resolve_ontology_entities",
        resolve or AsyncMock(return_value=([SimpleNamespace(name="frpm")], object())),
    )
    monkeypatch.setattr(_opb, "build", lambda *a, **k: "SYSTEM_PROMPT")
    monkeypatch.setattr(_sg.DataFabricGraph, "create", lambda **k: sentinel_graph)
    return sentinel_graph


async def test_ensure_graph_happy_path(monkeypatch):
    sentinel = _patch_ensure_graph_deps(monkeypatch)
    handler = DataFabricOntologyQueryHandler(
        ontologies=[("california-schools", "f1")], llm=MagicMock()
    )
    compiled = await handler._ensure_graph()
    assert compiled is sentinel
    # cached on second call (no re-work)
    assert await handler._ensure_graph() is sentinel


async def test_ensure_graph_owl_failure_degrades(monkeypatch):
    async def fetch(_es, _name, file_type, _fk):
        if file_type == "owl":
            raise RuntimeError("owl boom")
        return ("R2RML", "text/turtle")

    sentinel = _patch_ensure_graph_deps(monkeypatch, fetch=fetch)
    handler = DataFabricOntologyQueryHandler(
        ontologies=[("california-schools", "f1")], llm=MagicMock()
    )
    # R2RML is critical (present) so it still compiles; OWL failure degrades.
    assert await handler._ensure_graph() is sentinel


async def test_ensure_graph_no_entities_raises(monkeypatch):
    _patch_ensure_graph_deps(monkeypatch, parse=lambda _txt: [])
    handler = DataFabricOntologyQueryHandler(
        ontologies=[("california-schools", "f1")], llm=MagicMock()
    )
    with pytest.raises(ValueError, match="declared no entities"):
        await handler._ensure_graph()


async def test_ensure_graph_returns_cached_without_work(monkeypatch):
    handler = DataFabricOntologyQueryHandler(ontologies=[], llm=MagicMock())
    handler._compiled = "CACHED"  # type: ignore[assignment]
    # Flag intentionally OFF: if it did any work it would raise; cache short-circuits.
    monkeypatch.setattr(FeatureFlags, "is_flag_enabled", lambda *a, **k: False)
    assert await handler._ensure_graph() == "CACHED"


# --- __call__ (invoke + terminal message handling) --------------------------


def _handler_with_state(monkeypatch, messages):
    handler = DataFabricOntologyQueryHandler(ontologies=[], llm=MagicMock())
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"messages": messages})
    monkeypatch.setattr(handler, "_ensure_graph", AsyncMock(return_value=graph))
    return handler


async def test_call_collapses_terminal_tool_messages(monkeypatch):
    handler = _handler_with_state(
        monkeypatch,
        [AIMessage(content=""), ToolMessage(content="rows!", tool_call_id="c1")],
    )
    assert await handler("q") == "rows!"


async def test_call_returns_ai_message_content(monkeypatch):
    handler = _handler_with_state(monkeypatch, [AIMessage(content="the answer")])
    assert await handler("q") == "the answer"


async def test_call_fallback_when_no_answer(monkeypatch):
    handler = _handler_with_state(monkeypatch, [AIMessage(content="")])
    assert await handler("q") == "Unable to generate an answer from the available data."


async def test_ensure_graph_empty_resolution_raises(monkeypatch):
    # pairs is non-empty (parse) but resolution yields no entities -> post-resolve guard.
    _patch_ensure_graph_deps(monkeypatch, resolve=AsyncMock(return_value=([], object())))
    handler = DataFabricOntologyQueryHandler(
        ontologies=[("california-schools", "f1")], llm=MagicMock()
    )
    with pytest.raises(ValueError, match="could be resolved"):
        await handler._ensure_graph()


# --- terminal-message formatting --------------------------------------------


def test_format_terminal_collapses_multiple_results():
    out = DataFabricOntologyQueryHandler._format_terminal_tool_messages(
        [
            ToolMessage(content="rows-a", tool_call_id="1"),
            ToolMessage(content="rows-b", tool_call_id="2"),
        ]
    )
    assert "Multiple SQL queries" in out
    assert "Result 1:\nrows-a" in out
    assert "Result 2:\nrows-b" in out


def test_format_terminal_all_empty_returns_fallback():
    out = DataFabricOntologyQueryHandler._format_terminal_tool_messages(
        [ToolMessage(content="", tool_call_id="1")]
    )
    assert out == "Unable to generate an answer from the available data."
