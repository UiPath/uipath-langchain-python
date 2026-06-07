"""Tests for ContextGroundingRetriever's include_system_indexes plumbing."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from uipath.platform import UiPath

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
