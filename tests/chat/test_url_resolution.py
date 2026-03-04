"""Unit tests for resolve_gateway_url and inject_routing integration."""

import os
from unittest.mock import patch

import pytest

from uipath_langchain.chat.http_client.url import resolve_gateway_url

ENDPOINT = "agenthub_/llm/raw/vendor/openai/model/gpt-4/completions"


class TestResolveGatewayUrl:
    """Verify resolve_gateway_url returns correct URL and override flag."""

    def test_fallback_to_uipath_url(self) -> None:
        env = {"UIPATH_URL": "https://cloud.uipath.com/org/tenant"}
        with patch.dict(os.environ, env, clear=True):
            url, is_override = resolve_gateway_url(ENDPOINT)
        assert url == f"https://cloud.uipath.com/org/tenant/{ENDPOINT}"
        assert is_override is False

    def test_raises_when_no_url_available(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="UIPATH_URL"):
                resolve_gateway_url(ENDPOINT)

    def test_strips_trailing_slash_from_uipath_url(self) -> None:
        env = {"UIPATH_URL": "https://cloud.uipath.com/org/tenant/"}
        with patch.dict(os.environ, env, clear=True):
            url, is_override = resolve_gateway_url(ENDPOINT)
        assert url == f"https://cloud.uipath.com/org/tenant/{ENDPOINT}"
        assert is_override is False

    def test_service_override_used_when_available(self) -> None:
        """When resolve_service_url returns a URL, it's used with is_override=True."""
        with patch(
            "uipath_langchain.chat.http_client.url._resolve_service_url",
            return_value="http://localhost:8080/llm/raw/vendor/openai/model/gpt-4/completions",
        ):
            env = {"UIPATH_URL": "https://cloud.uipath.com/org/tenant"}
            with patch.dict(os.environ, env, clear=True):
                url, is_override = resolve_gateway_url(ENDPOINT)
        assert (
            url == "http://localhost:8080/llm/raw/vendor/openai/model/gpt-4/completions"
        )
        assert is_override is True

    def test_service_override_returns_none_falls_back(self) -> None:
        """When resolve_service_url returns None, fall back to UIPATH_URL."""
        with patch(
            "uipath_langchain.chat.http_client.url._resolve_service_url",
            return_value=None,
        ):
            env = {"UIPATH_URL": "https://cloud.uipath.com/org/tenant"}
            with patch.dict(os.environ, env, clear=True):
                url, is_override = resolve_gateway_url(ENDPOINT)
        assert url == f"https://cloud.uipath.com/org/tenant/{ENDPOINT}"
        assert is_override is False

    def test_no_override_for_non_prefixed_endpoint(self) -> None:
        env = {"UIPATH_URL": "https://cloud.uipath.com/org/tenant"}
        endpoint = "some/other/endpoint"
        with patch.dict(os.environ, env, clear=True):
            url, is_override = resolve_gateway_url(endpoint)
        assert url == f"https://cloud.uipath.com/org/tenant/{endpoint}"
        assert is_override is False


class TestRoutingHeadersInjection:
    """Verify build_uipath_headers injects routing headers when inject_routing=True."""

    def test_no_routing_headers_by_default(self) -> None:
        from uipath_langchain.chat.http_client import build_uipath_headers

        env = {
            "UIPATH_TENANT_ID": "tenant-abc",
            "UIPATH_ORGANIZATION_ID": "org-xyz",
        }
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_headers("fake-token")
        assert "X-UiPath-Internal-TenantId" not in headers
        assert "X-UiPath-Internal-AccountId" not in headers

    def test_routing_headers_injected_when_override(self) -> None:
        from uipath_langchain.chat.http_client import build_uipath_headers

        env = {
            "UIPATH_TENANT_ID": "tenant-abc",
            "UIPATH_ORGANIZATION_ID": "org-xyz",
        }
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_headers("fake-token", inject_routing=True)
        assert headers["X-UiPath-Internal-TenantId"] == "tenant-abc"
        assert headers["X-UiPath-Internal-AccountId"] == "org-xyz"

    def test_routing_headers_omitted_when_env_missing(self) -> None:
        from uipath_langchain.chat.http_client import build_uipath_headers

        with patch.dict(os.environ, {}, clear=True):
            headers = build_uipath_headers("fake-token", inject_routing=True)
        assert "X-UiPath-Internal-TenantId" not in headers
        assert "X-UiPath-Internal-AccountId" not in headers

    def test_partial_routing_headers(self) -> None:
        from uipath_langchain.chat.http_client import build_uipath_headers

        env = {"UIPATH_ORGANIZATION_ID": "org-only"}
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_headers("fake-token", inject_routing=True)
        assert "X-UiPath-Internal-TenantId" not in headers
        assert headers["X-UiPath-Internal-AccountId"] == "org-only"
