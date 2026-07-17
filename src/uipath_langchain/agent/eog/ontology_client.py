"""Async REST client for ontology-runtime."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs

from .graph_topology import OntologyGraph, parse_ofn


class OntologyClient:
    """Async HTTP client for the ontology-runtime service.

    Args:
        base_url: Root URL of the ontology-runtime (no trailing slash).
        account: Account name used in the API path.
        tenant: Tenant name used in the API path.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5002",
        account: str = "datafabric",
        tenant: str = "DefaultTenant",
        *,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.account = account
        self.tenant = tenant
        self.timeout = timeout

    @property
    def _api_base(self) -> str:
        """Build the base API path."""
        return (
            f"{self.base_url}/{self.account}/{self.tenant}"
            "/datafabric_/api"
        )

    def _make_client(self) -> httpx.AsyncClient:
        """Create a configured async HTTP client."""
        kwargs = get_httpx_client_kwargs()
        kwargs["timeout"] = self.timeout
        return httpx.AsyncClient(**kwargs)

    async def discover(self, ontology: str) -> dict[str, Any]:
        """Retrieve ontology metadata.

        Args:
            ontology: Name or ID of the deployed ontology.

        Returns:
            JSON response with ontology metadata.
        """
        url = f"{self._api_base}/ontology/{ontology}"
        async with self._make_client() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

    async def list_functions(self, ontology: str) -> list[dict[str, Any]]:
        """List function definitions for an ontology.

        Args:
            ontology: Name or ID of the deployed ontology.

        Returns:
            List of function definition dicts.
        """
        url = f"{self._api_base}/ontology/{ontology}/functions"
        async with self._make_client() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

    async def invoke_function(
        self,
        ontology: str,
        fn_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke an ontology function.

        Args:
            ontology: Name or ID of the deployed ontology.
            fn_name: Function name to invoke.
            params: Optional parameters to pass to the function.

        Returns:
            JSON response (typically ``{"rows": [...]}``)
        """
        url = (
            f"{self._api_base}/ontology/{ontology}"
            f"/functions/{fn_name}/invoke"
        )
        body: dict[str, Any] = {"params": params} if params else {}
        async with self._make_client() as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]

    async def fetch_artifact(self, ontology: str, filename: str) -> str:
        """Download a raw artifact file (OWL, SHACL, YARRRML, etc.).

        Args:
            ontology: Name or ID of the deployed ontology.
            filename: Artifact filename (e.g., ``schema.ofn``).

        Returns:
            Raw text content of the artifact.
        """
        url = (
            f"{self._api_base}/ontology/{ontology}"
            f"/artifact/{filename}"
        )
        async with self._make_client() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def fetch_graph(self, ontology: str) -> OntologyGraph:
        """Fetch and parse the ontology's entity-relationship graph.

        Downloads the OWL schema artifact, parses it into an
        ``OntologyGraph`` with nodes (entities), edges (relationships),
        and adjacency indexes. Also fetches function definitions and
        maps them to entities.

        Args:
            ontology: Name or ID of the deployed ontology.

        Returns:
            Parsed ``OntologyGraph`` ready for EoG traversal.
        """
        ofn_text, functions = await asyncio.gather(
            self.fetch_artifact(ontology, "schema.ofn"),
            self.list_functions(ontology),
        )
        graph = parse_ofn(ofn_text)
        graph.functions = functions
        graph._build_adjacency()
        return graph

    async def sparql(self, ontology: str, query: str) -> dict[str, Any]:
        """Execute a SPARQL query against an ontology.

        Args:
            ontology: Name or ID of the deployed ontology.
            query: Raw SPARQL query text.

        Returns:
            JSON response with query results.
        """
        url = f"{self._api_base}/ontology/{ontology}/sparql"
        async with self._make_client() as client:
            resp = await client.post(
                url,
                content=query,
                headers={"Content-Type": "application/sparql-query"},
            )
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
