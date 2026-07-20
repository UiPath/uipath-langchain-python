"""Tests for OntologyClient."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from uipath_langchain.agent.eog.ontology_client import OntologyClient


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    response_body: Any,
    status_code: int = 200,
    captured: list[httpx.Request] | None = None,
) -> None:
    """Patch OntologyClient._make_client to use a mock transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(
            status_code=status_code,
            json=response_body,
            request=request,
        )

    def _make_mock_client(self: OntologyClient) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(OntologyClient, "_make_client", _make_mock_client)


@pytest.fixture
def client() -> OntologyClient:
    return OntologyClient(
        base_url="http://test-host:5002",
        account="acct",
        tenant="tenant1",
    )


class TestApiBase:
    def test_api_base_path(self, client: OntologyClient) -> None:
        assert client._api_base == (
            "http://test-host:5002/acct/tenant1/datafabric_/api"
        )

    def test_trailing_slash_stripped(self) -> None:
        c = OntologyClient(base_url="http://host:5002/")
        assert not c._api_base.startswith("http://host:5002//")


class TestDiscover:
    @pytest.mark.asyncio
    async def test_discover_success(
        self,
        client: OntologyClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = {"name": "test-onto", "entities": ["e1"]}
        _patch_client(monkeypatch, expected)
        result = await client.discover("test-onto")
        assert result == expected


class TestListFunctions:
    @pytest.mark.asyncio
    async def test_list_functions_success(
        self,
        client: OntologyClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = [{"name": "fn1", "description": "desc1"}]
        _patch_client(monkeypatch, expected)
        result = await client.list_functions("test-onto")
        assert result == expected

    @pytest.mark.asyncio
    async def test_list_functions_with_touches_filter(
        self,
        client: OntologyClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = [{"name": "fn1", "touches": ["Invoice"]}]
        captured: list[httpx.Request] = []
        _patch_client(monkeypatch, expected, captured=captured)
        result = await client.list_functions("test-onto", touches="Invoice")
        assert result == expected
        assert "touches=Invoice" in str(captured[0].url)


class TestInvokeFunction:
    @pytest.mark.asyncio
    async def test_invoke_function_with_params(
        self,
        client: OntologyClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = {"rows": [{"id": "r1"}]}
        captured: list[httpx.Request] = []
        _patch_client(monkeypatch, expected, captured=captured)
        result = await client.invoke_function(
            "test-onto", "fn1", {"key": "val"}
        )
        assert result == expected
        body = json.loads(captured[0].content)
        assert body == {"params": {"key": "val"}}

    @pytest.mark.asyncio
    async def test_invoke_function_no_params(
        self,
        client: OntologyClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = {"rows": []}
        captured: list[httpx.Request] = []
        _patch_client(monkeypatch, expected, captured=captured)
        result = await client.invoke_function("test-onto", "fn1")
        assert result == expected
        body = json.loads(captured[0].content)
        assert body == {}


class TestSparql:
    @pytest.mark.asyncio
    async def test_sparql_success(
        self,
        client: OntologyClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = {"results": {"bindings": []}}
        captured: list[httpx.Request] = []
        _patch_client(monkeypatch, expected, captured=captured)
        result = await client.sparql(
            "test-onto", "SELECT ?s WHERE {?s ?p ?o}"
        )
        assert result == expected
        assert (
            captured[0].headers["content-type"] == "application/sparql-query"
        )


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_non_200_raises(
        self,
        client: OntologyClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, {"error": "fail"}, status_code=500)
        with pytest.raises(httpx.HTTPStatusError):
            await client.discover("test-onto")
