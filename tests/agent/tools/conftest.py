"""Shared fixtures for agent tool tests."""

import httpx
import pytest
from uipath.platform.errors import EnrichedException


@pytest.fixture
def make_enriched_exception():
    """Factory fixture for creating EnrichedException with a given status code and URL."""

    def _make(
        status_code: int,
        body: str = "Bad Request",
        url: str = "https://cloud.uipath.com/test_/endpoint",
    ) -> EnrichedException:
        response = httpx.Response(
            status_code=status_code,
            content=body.encode(),
            request=httpx.Request("POST", url),
        )
        http_error = httpx.HTTPStatusError(
            message=body,
            request=response.request,
            response=response,
        )
        return EnrichedException(http_error)

    return _make
