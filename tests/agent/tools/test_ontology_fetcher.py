"""Tests for ontology fetching (datafabric_tool/ontology_fetcher.py)."""

from unittest.mock import AsyncMock, MagicMock

from uipath_langchain.agent.tools.datafabric_tool import ontology_fetcher
from uipath_langchain.agent.tools.datafabric_tool.ontology_fetcher import (
    _notation_label,
    fetch_ontology_text,
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


# --- fetch_ontology_text ---------------------------------------------------


async def test_no_ontologies_returns_empty():
    assert await fetch_ontology_text(_entities_service(), []) == ""


async def test_single_ontology_returns_fenced_block():
    es = _entities_service(content="OWLBODY", media_type="text/turtle")

    result = await fetch_ontology_text(es, [("library", "folder-1")])

    assert "ONTOLOGY: library" in result
    assert "OWLBODY" in result
    assert "Turtle" in result
    es.get_ontology_file_async.assert_awaited_once_with("library", "owl", "folder-1")


async def test_multiple_ontologies_concatenated():
    es = _entities_service()

    result = await fetch_ontology_text(es, [("library", None), ("finance", "f2")])

    assert "ONTOLOGY: library" in result
    assert "ONTOLOGY: finance" in result
    assert es.get_ontology_file_async.await_count == 2


async def test_graceful_degrade_on_error():
    es = MagicMock()
    es.get_ontology_file_async = AsyncMock(side_effect=RuntimeError("boom"))

    result = await fetch_ontology_text(es, [("library", None)])

    assert "unavailable" in result
    assert "RuntimeError" in result  # the exception type is surfaced, not raised


async def test_oversized_owl_is_degraded(monkeypatch):
    monkeypatch.setattr(ontology_fetcher, "_MAX_OWL_BYTES", 5)
    es = _entities_service(content="0123456789")  # 10 bytes > cap

    result = await fetch_ontology_text(es, [("library", None)])

    assert "unavailable" in result
