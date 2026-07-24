"""Tests for http_request_tool.py module."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentInternalHttpRequestToolProperties,
    AgentInternalToolResourceConfig,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.internal_tools import http_request_tool as H
from uipath_langchain.agent.tools.internal_tools.http_request_tool import (
    create_http_request_tool,
)

# Patch mockable to a passthrough for every test in this module, matching the
# convention in the sibling internal-tool tests.
pytestmark = pytest.mark.usefixtures("_passthrough_mockable")


@pytest.fixture
def _passthrough_mockable():
    with patch(
        "uipath_langchain.agent.tools.internal_tools.http_request_tool.mockable",
        lambda **kwargs: lambda f: f,
    ):
        yield


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.fixture
def resource_config():
    # The input schema is authored in the agent definition (like analyze-files);
    # the tool reads these fields from kwargs rather than injecting them.
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string"},
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "params": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "body": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "object", "additionalProperties": True},
                ]
            },
            "timeout": {"type": "number"},
        },
        "required": ["url"],
    }
    return AgentInternalToolResourceConfig(
        name="http_request",
        description="Make an HTTP request",
        input_schema=input_schema,
        output_schema={},
        properties=AgentInternalHttpRequestToolProperties(),
    )


class TestCreateHttpRequestTool:
    def test_tool_creation_and_schema(self, resource_config, mock_llm):
        """Tool is created with the canonical input fields and fixed output fields."""
        tool = create_http_request_tool(resource_config, mock_llm)

        assert tool.name == "http_request"
        assert tool.description == "Make an HTTP request"

        input_fields = set(tool.args_schema.model_fields.keys())
        assert {"url", "method", "headers", "params", "body", "timeout"} <= input_fields

        output_fields = set(tool.output_type.model_fields.keys())
        assert output_fields == {"statusCode", "headers", "body"}

    @patch(
        "uipath_langchain.agent.tools.internal_tools.http_request_tool._assert_public_url",
        new_callable=AsyncMock,
    )
    async def test_get_request_happy_path(
        self, _mock_guard, httpx_mock, resource_config, mock_llm
    ):
        """A GET request returns statusCode/headers/body; method is normalized."""
        httpx_mock.add_response(
            url="https://api.example.com/items?page=1",
            method="GET",
            status_code=200,
            headers={"Content-Type": "application/json"},
            text='{"ok": true}',
        )

        tool = create_http_request_tool(resource_config, mock_llm)
        result = await tool.coroutine(
            url="https://api.example.com/items",
            method="get",  # lowercase -> normalized to GET
            params={"page": "1"},
        )

        assert result["statusCode"] == 200
        assert result["headers"]["content-type"] == "application/json"
        assert result["body"] == '{"ok": true}'

    @patch(
        "uipath_langchain.agent.tools.internal_tools.http_request_tool._assert_public_url",
        new_callable=AsyncMock,
    )
    async def test_dict_body_is_sent_as_json(
        self, _mock_guard, httpx_mock, resource_config, mock_llm
    ):
        """An object body is serialized as JSON."""
        httpx_mock.add_response(
            url="https://api.example.com/create",
            method="POST",
            status_code=201,
            text="created",
        )

        tool = create_http_request_tool(resource_config, mock_llm)
        result = await tool.coroutine(
            url="https://api.example.com/create",
            method="POST",
            headers={"X-Trace": "abc"},
            body={"name": "widget", "qty": 3},
        )

        assert result["statusCode"] == 201
        request = httpx_mock.get_requests()[0]
        assert request.headers["x-trace"] == "abc"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.read()) == {"name": "widget", "qty": 3}

    @patch(
        "uipath_langchain.agent.tools.internal_tools.http_request_tool._assert_public_url",
        new_callable=AsyncMock,
    )
    async def test_string_body_is_sent_raw(
        self, _mock_guard, httpx_mock, resource_config, mock_llm
    ):
        """A string body is sent as-is."""
        httpx_mock.add_response(method="PUT", status_code=200, text="ok")

        tool = create_http_request_tool(resource_config, mock_llm)
        await tool.coroutine(
            url="https://api.example.com/raw",
            method="PUT",
            body="raw-payload",
        )

        assert httpx_mock.get_requests()[0].read() == b"raw-payload"

    @patch(
        "uipath_langchain.agent.tools.internal_tools.http_request_tool._assert_public_url",
        new_callable=AsyncMock,
    )
    async def test_non_2xx_is_returned_not_raised(
        self, _mock_guard, httpx_mock, resource_config, mock_llm
    ):
        """A 4xx/5xx response is returned to the agent rather than raised."""
        httpx_mock.add_response(status_code=404, text="not found")

        tool = create_http_request_tool(resource_config, mock_llm)
        result = await tool.coroutine(url="https://api.example.com/missing")

        assert result["statusCode"] == 404
        assert result["body"] == "not found"

    @patch(
        "uipath_langchain.agent.tools.internal_tools.http_request_tool._assert_public_url",
        new_callable=AsyncMock,
    )
    async def test_schemeless_url_defaults_to_https(
        self, _mock_guard, httpx_mock, resource_config, mock_llm
    ):
        """A URL without a scheme (e.g. 'google.com') is requested over https."""
        httpx_mock.add_response(
            url="https://google.com", method="GET", status_code=200, text="ok"
        )

        tool = create_http_request_tool(resource_config, mock_llm)
        result = await tool.coroutine(url="google.com")

        assert result["statusCode"] == 200
        assert str(httpx_mock.get_requests()[0].url) == "https://google.com"

    async def test_missing_url_raises(self, resource_config, mock_llm):
        tool = create_http_request_tool(resource_config, mock_llm)
        with pytest.raises(AgentRuntimeError, match="Argument 'url' is required"):
            await tool.coroutine()

    async def test_invalid_method_raises(self, resource_config, mock_llm):
        tool = create_http_request_tool(resource_config, mock_llm)
        with pytest.raises(AgentRuntimeError, match="Unsupported HTTP method"):
            await tool.coroutine(url="https://api.example.com/x", method="FETCH")

    async def test_ssrf_blocks_internal_host(self, resource_config, mock_llm):
        """A request to an internal address is blocked before it is issued."""
        tool = create_http_request_tool(resource_config, mock_llm)
        with pytest.raises(AgentRuntimeError):
            await tool.coroutine(url="http://127.0.0.1/admin")


class TestSsrfGuard:
    """Direct tests of the SSRF guard (also runs on every redirect hop)."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/x",
            "http://localhost:8080/",
            "http://10.0.0.5/",
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/",
            "ftp://example.com/file",
        ],
    )
    async def test_blocked_urls(self, url):
        with pytest.raises(AgentRuntimeError):
            await H._assert_public_url(url)

    async def test_public_url_allowed(self):
        # Resolves a real public host; should not raise.
        await H._assert_public_url("https://example.com/")


class TestNormalizeUrl:
    """Scheme defaulting: bare hosts become https, explicit schemes untouched."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("google.com", "https://google.com"),
            ("google.com/search?q=1", "https://google.com/search?q=1"),
            ("  google.com  ", "https://google.com"),
            ("//google.com", "https://google.com"),
            ("http://google.com", "http://google.com"),
            ("https://google.com", "https://google.com"),
            # An explicit non-HTTP scheme is preserved (SSRF guard rejects it later).
            ("ftp://host/file", "ftp://host/file"),
        ],
    )
    def test_normalize_url(self, raw, expected):
        assert H._normalize_url(raw) == expected
