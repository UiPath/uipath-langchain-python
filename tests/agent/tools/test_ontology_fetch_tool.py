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


def _typed_entities_service(
    owl: str = "OWLBODY", r2rml: str | None = "R2RMLBODY"
) -> MagicMock:
    """Entities service that returns distinct OWL/R2RML content per file_type.

    ``r2rml=None`` simulates an ontology with no R2RML mapping (the SDK raises).
    """
    es = MagicMock()

    async def _fake(name, file_type, folder_key=None):
        if file_type == "owl":
            return {"content": owl, "mediaType": "text/turtle"}
        if r2rml is None:
            raise FileNotFoundError("no r2rml file")
        return {"content": r2rml, "mediaType": "application/r2rml+turtle"}

    es.get_ontology_file_async = AsyncMock(side_effect=_fake)
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
    # Both the OWL schema and the R2RML mapping are requested for the ontology.
    es.get_ontology_file_async.assert_any_await("library", "owl", "folder-1")
    es.get_ontology_file_async.assert_any_await("library", "r2rml", "folder-1")
    assert es.get_ontology_file_async.await_count == 2


async def test_fetcher_includes_r2rml_when_present():
    es = _typed_entities_service(owl="OWLBODY", r2rml="R2RMLBODY")
    fetcher = OntologyFetcher(es, [("library", "f1")])

    result = await fetcher()

    assert "ONTOLOGY: library" in result and "OWLBODY" in result
    assert "R2RML MAPPING: library" in result and "R2RMLBODY" in result
    requested = {call.args[1] for call in es.get_ontology_file_async.await_args_list}
    assert requested == {"owl", "r2rml"}


async def test_fetcher_skips_absent_r2rml_without_warning():
    es = _typed_entities_service(owl="OWLBODY", r2rml=None)
    fetcher = OntologyFetcher(es, [("library", None)])

    result = await fetcher()

    assert "ONTOLOGY: library" in result  # OWL still present
    assert "R2RML" not in result  # absent optional mapping → no block
    assert "unavailable" not in result  # and no loud fallback for the optional file


async def test_fetcher_multiple_ontologies_concatenated():
    es = _entities_service()
    fetcher = OntologyFetcher(es, [("library", None), ("finance", "f2")])

    result = await fetcher()

    assert "ONTOLOGY: library" in result
    assert "ONTOLOGY: finance" in result
    # 2 ontologies x 2 file types (owl + r2rml).
    assert es.get_ontology_file_async.await_count == 4


async def test_fetcher_caches_after_first_call():
    es = _entities_service()
    fetcher = OntologyFetcher(es, [("library", None), ("finance", None)])

    first = await fetcher()
    second = await fetcher()

    assert first == second
    # Two ontologies x two file types, fetched once total — the second call is
    # served from cache.
    assert es.get_ontology_file_async.await_count == 4


async def test_fetcher_graceful_degrade_on_error():
    es = MagicMock()
    es.get_ontology_file_async = AsyncMock(side_effect=RuntimeError("boom"))
    fetcher = OntologyFetcher(es, [("library", None)])

    result = await fetcher()

    assert "unavailable" in result
    assert "RuntimeError" in result  # the exception type is surfaced, not raised


async def test_fetcher_oversized_owl_is_degraded(monkeypatch):
    monkeypatch.setattr(oft, "_MAX_FILE_BYTES", 5)
    es = _entities_service(content="0123456789")  # 10 bytes > cap
    fetcher = OntologyFetcher(es, [("library", None)])

    result = await fetcher()

    assert "unavailable" in result


# --- create_ontology_fetch_tool --------------------------------------------


def test_create_tool_metadata_and_schema():
    tool = create_ontology_fetch_tool(
        _entities_service(), [("library", None), ("finance", None)]
    )

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
