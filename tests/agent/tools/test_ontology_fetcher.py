"""Tests for ontology file fetching (datafabric_tool/ontology_fetcher.py)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from uipath_langchain.agent.tools.datafabric_tool.ontology import ontology_fetcher
from uipath_langchain.agent.tools.datafabric_tool.ontology.ontology_fetcher import (
    fence_ontology_block,
    fetch_ontology_file,
)


def _entities_service(content: str = "BODY", media_type: str = "text/turtle"):
    es = MagicMock()
    es.get_ontology_file_async = AsyncMock(
        return_value={"content": content, "mediaType": media_type}
    )
    return es


# --- fetch_ontology_file ----------------------------------------------------


async def test_fetch_returns_content_and_media_type():
    es = _entities_service(content="OWLBODY", media_type="application/owl-functional")

    content, media_type = await fetch_ontology_file(es, "library", "owl", "folder-1")

    assert content == "OWLBODY"
    assert media_type == "application/owl-functional"
    es.get_ontology_file_async.assert_awaited_once_with("library", "owl", "folder-1")


async def test_fetch_r2rml_file_type_is_passed_through():
    es = _entities_service(content="@prefix rr: <> .")

    await fetch_ontology_file(es, "library", "r2rml", None)

    es.get_ontology_file_async.assert_awaited_once_with("library", "r2rml", None)


async def test_fetch_raises_on_underlying_error():
    es = MagicMock()
    es.get_ontology_file_async = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await fetch_ontology_file(es, "library", "r2rml", None)


async def test_fetch_raises_when_oversized(monkeypatch):
    monkeypatch.setattr(ontology_fetcher, "_MAX_ONTOLOGY_BYTES", 5)
    es = _entities_service(content="0123456789")  # 10 bytes > cap

    with pytest.raises(ValueError, match="size limit"):
        await fetch_ontology_file(es, "library", "owl", None)


async def test_fetch_missing_content_defaults_empty():
    es = MagicMock()
    es.get_ontology_file_async = AsyncMock(return_value={})

    content, media_type = await fetch_ontology_file(es, "library", "owl", None)

    assert content == ""
    assert media_type == ""


# --- fence_ontology_block ---------------------------------------------------


def test_fence_includes_type_name_and_media_type():
    block = fence_ontology_block("library", "owl", "CONTENT", "text/turtle")

    assert block.startswith("--- OWL, text/turtle: library ---")
    assert "CONTENT" in block
    assert block.endswith("--- END OWL: library ---")


def test_fence_without_media_type():
    block = fence_ontology_block("library", "r2rml", "MAP")

    assert block.startswith("--- R2RML: library ---")
    assert "MAP" in block
    assert block.endswith("--- END R2RML: library ---")
