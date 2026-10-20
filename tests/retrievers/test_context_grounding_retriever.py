"""Tests for ContextGroundingRetriever's search plumbing and document metadata."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from uipath.platform import UiPath
from uipath.platform.context_grounding import (
    ContextGroundingSearchResultItem,
    SemanticSearchResult,
    UnifiedQueryResult,
)

from uipath_langchain.retrievers import ContextGroundingRetriever


def _make_unified_search_result() -> MagicMock:
    result = MagicMock()
    result.semantic_results.values = []
    result.semantic_results.metadata.operation_id = "op-1"
    return result


def _make_sdk_mock() -> MagicMock:
    return MagicMock(spec=UiPath)


def test_retriever_forwards_include_system_indexes_when_true() -> None:
    sdk = _make_sdk_mock()
    sdk.context_grounding.unified_search.return_value = _make_unified_search_result()

    retriever = ContextGroundingRetriever(
        index_name="my-index",
        uipath_sdk=sdk,
        include_system_indexes=True,
    )
    retriever.invoke("hello")

    sdk.context_grounding.unified_search.assert_called_once()
    kwargs = sdk.context_grounding.unified_search.call_args.kwargs
    assert kwargs["include_system_indexes"] is True


def test_retriever_defaults_include_system_indexes_to_false() -> None:
    sdk = _make_sdk_mock()
    sdk.context_grounding.unified_search.return_value = _make_unified_search_result()

    retriever = ContextGroundingRetriever(index_name="my-index", uipath_sdk=sdk)
    retriever.invoke("hello")

    kwargs = sdk.context_grounding.unified_search.call_args.kwargs
    assert kwargs["include_system_indexes"] is False


@pytest.mark.asyncio
async def test_retriever_async_forwards_include_system_indexes_when_true() -> None:
    sdk = _make_sdk_mock()
    sdk.context_grounding.unified_search_async = AsyncMock(
        return_value=_make_unified_search_result()
    )

    retriever = ContextGroundingRetriever(
        index_name="my-index",
        uipath_sdk=sdk,
        include_system_indexes=True,
    )
    await retriever.ainvoke("hello")

    sdk.context_grounding.unified_search_async.assert_awaited_once()
    kwargs = sdk.context_grounding.unified_search_async.call_args.kwargs
    assert kwargs["include_system_indexes"] is True


@pytest.mark.asyncio
async def test_retriever_async_defaults_include_system_indexes_to_false() -> None:
    sdk = _make_sdk_mock()
    sdk.context_grounding.unified_search_async = AsyncMock(
        return_value=_make_unified_search_result()
    )

    retriever = ContextGroundingRetriever(index_name="my-index", uipath_sdk=sdk)
    await retriever.ainvoke("hello")

    kwargs = sdk.context_grounding.unified_search_async.call_args.kwargs
    assert kwargs["include_system_indexes"] is False


def _make_result_with_one_hit() -> UnifiedQueryResult:
    return UnifiedQueryResult(
        semantic_results=SemanticSearchResult(
            values=[
                ContextGroundingSearchResultItem(
                    content="Total revenue",
                    source="report.pdf",
                    page_number=7,
                    source_document_id="doc-1",
                )
            ]
        )
    )


def test_retriever_keeps_the_source_document_id() -> None:
    sdk = _make_sdk_mock()
    sdk.context_grounding.unified_search.return_value = _make_result_with_one_hit()

    retriever = ContextGroundingRetriever(index_name="my-index", uipath_sdk=sdk)
    docs = retriever.invoke("revenue")

    assert docs[0].metadata["source_document_id"] == "doc-1"
    assert docs[0].metadata["page_number"] == 7


@pytest.mark.asyncio
async def test_retriever_async_keeps_the_source_document_id() -> None:
    sdk = _make_sdk_mock()
    sdk.context_grounding.unified_search_async = AsyncMock(
        return_value=_make_result_with_one_hit()
    )

    retriever = ContextGroundingRetriever(index_name="my-index", uipath_sdk=sdk)
    docs = await retriever.ainvoke("revenue")

    assert docs[0].metadata["source_document_id"] == "doc-1"
