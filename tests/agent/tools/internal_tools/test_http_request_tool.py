"""Tests for http_request_tool.py module.

These tests exercise the tool only through its public surface (the created
tool's ``coroutine``/``ainvoke``), never its module-private helpers. Requests
target a literal public IP (``8.8.8.8``) so the real SSRF guard runs and passes
without any DNS lookup — no need to mock internal functions.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentInternalHttpRequestToolProperties,
    AgentInternalToolResourceConfig,
)

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.internal_tools.http_request_tool import (
    create_http_request_tool,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

# A literal public IP: the SSRF guard treats it as public, and getaddrinfo on a
# numeric host resolves locally, so tests need no network and no patching.
PUBLIC_HOST = "8.8.8.8"
BASE_URL = f"https://{PUBLIC_HOST}"

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
    # Headers/params are arrays of name/value pairs (not open maps) so the
    # generated tool schema stays compatible with providers that reject
    # additionalProperties (e.g. Gemini); body is a plain string.
    pair_list = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["name", "value"],
        },
    }
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string"},
            "headers": pair_list,
            "params": pair_list,
            "body": {"type": "string"},
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


@pytest.fixture
def tool(resource_config, mock_llm):
    return create_http_request_tool(resource_config, mock_llm)


class TestCreateHttpRequestTool:
    def test_tool_creation_and_schema(self, tool):
        """Tool is created with the canonical input fields and fixed output fields."""
        assert isinstance(tool, StructuredToolWithArgumentProperties)
        assert tool.name == "http_request"
        assert tool.description == "Make an HTTP request"

        args_schema = tool.args_schema
        assert isinstance(args_schema, type) and issubclass(args_schema, BaseModel)
        input_fields = set(args_schema.model_fields.keys())
        assert {"url", "method", "headers", "params", "body", "timeout"} <= input_fields

        output_fields = set(tool.output_type.model_fields.keys())
        assert output_fields == {"statusCode", "headers", "body"}

    async def test_get_request_happy_path(self, tool, httpx_mock):
        """A GET request returns statusCode/headers/body; method is normalized."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/items?page=1",
            method="GET",
            status_code=200,
            headers={"Content-Type": "application/json"},
            text='{"ok": true}',
        )

        result = await tool.ainvoke(
            {
                "url": f"{BASE_URL}/items",
                "method": "get",  # lowercase -> normalized to GET
                "params": [{"name": "page", "value": "1"}],
            }
        )

        assert result["statusCode"] == 200
        assert result["headers"]["content-type"] == "application/json"
        assert result["body"] == '{"ok": true}'

    async def test_json_string_body_gets_default_content_type(self, tool, httpx_mock):
        """A JSON object/array string body is sent verbatim, with an auto Content-Type."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/create", method="POST", status_code=201, text="created"
        )

        result = await tool.ainvoke(
            {
                "url": f"{BASE_URL}/create",
                "method": "POST",
                "body": '{"name": "widget", "qty": 3}',  # no Content-Type header set
            }
        )

        assert result["statusCode"] == 201
        request = httpx_mock.get_requests()[0]
        assert request.headers["content-type"] == "application/json"
        assert request.read() == b'{"name": "widget", "qty": 3}'  # not re-serialized

    async def test_explicit_content_type_is_not_overridden(self, tool, httpx_mock):
        """A caller-set Content-Type wins over the JSON default."""
        httpx_mock.add_response(method="POST", status_code=200, text="ok")

        await tool.ainvoke(
            {
                "url": f"{BASE_URL}/create",
                "method": "POST",
                "headers": [{"name": "content-type", "value": "text/plain"}],
                "body": '{"a": 1}',
            }
        )

        assert httpx_mock.get_requests()[0].headers["content-type"] == "text/plain"

    async def test_non_json_string_body_has_no_default_content_type(
        self, tool, httpx_mock
    ):
        """A non-JSON string body is sent as-is with no Content-Type inferred."""
        httpx_mock.add_response(method="PUT", status_code=200, text="ok")

        await tool.ainvoke(
            {"url": f"{BASE_URL}/raw", "method": "PUT", "body": "raw-payload"}
        )

        request = httpx_mock.get_requests()[0]
        assert request.read() == b"raw-payload"
        assert "content-type" not in request.headers

    async def test_scalar_json_body_is_not_labeled_json(self, tool, httpx_mock):
        """A bare JSON scalar string (e.g. '123') is not treated as a JSON body."""
        httpx_mock.add_response(method="POST", status_code=200, text="ok")

        await tool.ainvoke({"url": f"{BASE_URL}/x", "method": "POST", "body": "123"})

        assert "content-type" not in httpx_mock.get_requests()[0].headers

    async def test_header_and_param_pairs_are_folded_into_request(
        self, tool, httpx_mock
    ):
        """Name/value pair lists are folded into httpx headers and query params.

        Pairs go through args-schema validation (arriving as pydantic models, as
        in the agent graph), exercising the list-of-pairs -> dict conversion.
        """
        httpx_mock.add_response(method="GET", status_code=200, text="ok")

        result = await tool.ainvoke(
            {
                "url": f"{BASE_URL}/x",
                "params": [
                    {"name": "a", "value": "3"},
                    {"name": "b", "value": "5"},
                ],
                "headers": [{"name": "X-Count", "value": "7"}],
            }
        )

        assert result["statusCode"] == 200
        request = httpx_mock.get_requests()[0]
        assert dict(request.url.params) == {"a": "3", "b": "5"}
        assert request.headers["x-count"] == "7"

    async def test_non_2xx_is_returned_not_raised(self, tool, httpx_mock):
        """A 4xx/5xx response is returned to the agent rather than raised."""
        httpx_mock.add_response(status_code=404, text="not found")

        result = await tool.ainvoke({"url": f"{BASE_URL}/missing"})

        assert result["statusCode"] == 404
        assert result["body"] == "not found"

    async def test_schemeless_url_defaults_to_https(self, tool, httpx_mock):
        """A URL without a scheme is requested over https."""
        httpx_mock.add_response(url=BASE_URL, method="GET", status_code=200, text="ok")

        result = await tool.ainvoke({"url": PUBLIC_HOST})  # no scheme

        assert result["statusCode"] == 200
        assert str(httpx_mock.get_requests()[0].url) == BASE_URL

    async def test_missing_url_raises(self, mock_llm):
        """The tool's own guard rejects a missing url.

        Uses a schema that does not mark ``url`` required, so args validation
        passes and the tool's defensive check is what raises (a schema that
        requires ``url`` would be rejected earlier, by validation).
        """
        resource = AgentInternalToolResourceConfig(
            name="http_request",
            description="Make an HTTP request",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}},
            output_schema={},
            properties=AgentInternalHttpRequestToolProperties(),
        )
        tool = create_http_request_tool(resource, mock_llm)
        with pytest.raises(AgentRuntimeError, match="Argument 'url' is required"):
            await tool.ainvoke({})

    async def test_non_string_url_raises(self, mock_llm):
        """A non-string url is rejected with a clean error.

        Uses a schema where ``url`` is untyped so validation passes a number
        through to the tool's own type guard.
        """
        resource = AgentInternalToolResourceConfig(
            name="http_request",
            description="Make an HTTP request",
            input_schema={"type": "object", "properties": {"url": {}}},
            output_schema={},
            properties=AgentInternalHttpRequestToolProperties(),
        )
        tool = create_http_request_tool(resource, mock_llm)
        with pytest.raises(AgentRuntimeError, match="'url' must be a string"):
            await tool.ainvoke({"url": 123})

    async def test_invalid_method_raises(self, tool):
        with pytest.raises(AgentRuntimeError, match="Unsupported HTTP method"):
            await tool.ainvoke({"url": f"{BASE_URL}/x", "method": "FETCH"})

    @pytest.mark.parametrize("bad_timeout", [-1, 0, -0.5])
    async def test_non_positive_timeout_raises(self, tool, bad_timeout):
        with pytest.raises(AgentRuntimeError, match="must be a positive number"):
            await tool.ainvoke({"url": f"{BASE_URL}/x", "timeout": bad_timeout})

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/admin",
            "http://localhost:8080/",
            "http://10.0.0.5/",
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/",
            f"ftp://{PUBLIC_HOST}/file",
        ],
    )
    async def test_ssrf_blocked_targets_are_rejected(self, tool, url):
        """Internal/metadata/non-http targets are rejected before any request.

        No response is registered because the request never leaves the SSRF
        guard.
        """
        with pytest.raises(AgentRuntimeError):
            await tool.ainvoke({"url": url})
