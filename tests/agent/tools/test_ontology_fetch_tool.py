"""Tests for the ontology fetch tool (datafabric_tool/ontology_fetch_tool.py)."""

from unittest.mock import AsyncMock, MagicMock

from uipath_langchain.agent.tools.datafabric_tool import ontology_fetch_tool as oft
from uipath_langchain.agent.tools.datafabric_tool.models import OntologyFetchInput
from uipath_langchain.agent.tools.datafabric_tool.ontology_fetch_tool import (
    OntologyFetcher,
    _notation_label,
    create_ontology_fetch_tool,
)


def _entities_service(content: str = "OWLDATA", media_type: str = "text/turtle"):
    es = MagicMock()
    es.get_ontology_file_async = AsyncMock(
        return_value={"content": content, "mediaType": media_type}
    )
    return es


# --- _notation_label -------------------------------------------------------


def test_notation_label_turtle():
    assert _notation_label("text/turtle") == "Turtle"
    assert _notation_label("application/ttl") == "Turtle"


def test_notation_label_functional():
    assert _notation_label("application/owl-functional") == "OWL Functional Notation"
    assert _notation_label("text/ofn") == "OWL Functional Notation"


def test_notation_label_unknown_defaults():
    assert _notation_label("") == "Turtle or OWL Functional Notation"
    assert _notation_label("application/json") == "Turtle or OWL Functional Notation"


# --- OntologyFetchInput ----------------------------------------------------


def test_ontology_fetch_input_is_empty():
    # Intentionally empty: the name is pinned from config, never the LLM.
    assert OntologyFetchInput().model_dump() == {}


# --- OntologyFetcher -------------------------------------------------------


async def test_fetcher_no_ontologies_returns_message():
    fetcher = OntologyFetcher(_entities_service(), [])
    result = await fetcher()
    assert "No ontologies are configured" in result


async def test_fetcher_single_ontology_returns_fenced_block():
    es = _entities_service(content="OWLBODY", media_type="text/turtle")
    fetcher = OntologyFetcher(es, [("library", "folder-1")])

    result = await fetcher()

    assert "ONTOLOGY: library" in result
    assert "OWLBODY" in result
    assert "Turtle" in result
    es.get_ontology_file_async.assert_awaited_once_with("library", "owl", "folder-1")


async def test_fetcher_multiple_ontologies_concatenated():
    es = _entities_service()
    fetcher = OntologyFetcher(es, [("library", None), ("finance", "f2")])

    result = await fetcher()

    assert "ONTOLOGY: library" in result
    assert "ONTOLOGY: finance" in result
    assert es.get_ontology_file_async.await_count == 2


async def test_fetcher_caches_after_first_call():
    es = _entities_service()
    fetcher = OntologyFetcher(es, [("library", None), ("finance", None)])

    first = await fetcher()
    second = await fetcher()

    assert first == second
    # Two ontologies fetched once total — the second call is served from cache.
    assert es.get_ontology_file_async.await_count == 2


async def test_fetcher_graceful_degrade_on_error():
    es = MagicMock()
    es.get_ontology_file_async = AsyncMock(side_effect=RuntimeError("boom"))
    fetcher = OntologyFetcher(es, [("library", None)])

    result = await fetcher()

    assert "unavailable" in result
    assert "RuntimeError" in result  # the exception type is surfaced, not raised


async def test_fetcher_oversized_owl_is_degraded(monkeypatch):
    monkeypatch.setattr(oft, "_MAX_OWL_BYTES", 5)
    es = _entities_service(content="0123456789")  # 10 bytes > cap
    fetcher = OntologyFetcher(es, [("library", None)])

    result = await fetcher()

    assert "unavailable" in result


# --- create_ontology_fetch_tool --------------------------------------------


def test_create_tool_metadata_and_schema():
    tool = create_ontology_fetch_tool(_entities_service(), [("library", None), ("finance", None)])

    assert tool.name == "fetch_ontology"
    assert "library" in tool.description and "finance" in tool.description
    assert tool.args_schema is OntologyFetchInput
    assert tool.metadata == {"tool_type": "ontology_fetch"}


async def test_create_tool_invocation_fetches_ontology():
    es = _entities_service(content="OWLBODY")
    tool = create_ontology_fetch_tool(es, [("library", None)])

    result = await tool.ainvoke({})

    assert "library" in result
    assert "OWLBODY" in result
