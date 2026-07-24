"""Internal HTTP request tool.

Issues an outbound HTTP request to a caller-provided URL. The tool's arguments
(``url``/``method``/``headers``/``params``/``body``/``timeout``) come from the
resource's input schema authored in the agent definition, mirroring the
analyze-files tool; each may be pinned to a static value, bound to an agent
input, or left for the LLM to infer through the generic ``argument_properties``
mechanism shared by all structured tools.

Requests that resolve to private, loopback, link-local, or cloud-metadata
addresses are rejected to guard against SSRF; the check runs on every request,
including redirect hops.
"""

import asyncio
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.agent.models.agent import AgentInternalToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model,
    create_output_model,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

# HTTP methods the http-request tool supports.
HTTP_REQUEST_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]

# Default per-request timeout (seconds) applied when neither the configured
# argument properties nor the LLM provide one.
HTTP_REQUEST_DEFAULT_TIMEOUT_SECONDS = 30.0

# Fixed output schema for the http-request tool.
HTTP_REQUEST_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statusCode": {
            "type": "integer",
            "description": "HTTP status code of the response.",
        },
        "headers": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Response headers.",
        },
        "body": {
            "type": "string",
            "description": "Response body as text.",
        },
    },
    "required": ["statusCode", "headers", "body"],
}

# Maximum number of redirects to follow. Each hop is re-validated by the SSRF
# guard, so this only bounds how long a redirect chain may run.
_MAX_REDIRECTS = 5

# Matches a leading URI scheme (e.g. ``https://``, ``ftp://``). Used to detect
# whether the caller supplied a scheme at all.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def _normalize_url(url: str) -> str:
    """Default to ``https`` when the caller omits a scheme.

    ``"google.com"`` becomes ``"https://google.com"``; a scheme-relative
    ``"//google.com"`` becomes ``"https://google.com"``. A URL that already
    carries an explicit scheme is returned unchanged — including non-HTTP(S)
    schemes, which the SSRF guard then rejects.
    """
    url = url.strip()
    if url.startswith("//"):
        return f"https:{url}"
    if not _SCHEME_RE.match(url):
        return f"https://{url}"
    return url


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an IP is non-public (private, loopback, link-local, ...).

    Link-local covers the cloud-metadata address ``169.254.169.254``.
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _blocked_host_error(detail: str) -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.HTTP_ERROR,
        title="Request was blocked",
        detail=detail,
        category=UiPathErrorCategory.USER,
    )


async def _assert_public_url(url: str) -> None:
    """Reject a URL whose host is missing, non-HTTP(S), or internal.

    Resolves the host and fails if *any* resolved address is non-public, so a
    hostname that maps to an internal IP cannot slip through.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise _blocked_host_error(
            f"Unsupported URL scheme {parsed.scheme!r}; only http and https are allowed."
        )

    host = parsed.hostname
    if not host:
        raise _blocked_host_error(f"URL {url!r} has no host.")

    # A bare IP literal in the URL still needs checking.
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and _is_blocked_ip(literal_ip):
        raise _blocked_host_error(f"Host {host!r} resolves to a non-public address.")

    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, parsed.port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as e:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.HTTP_ERROR,
            title="Host could not be resolved",
            detail=f"Could not resolve host {host!r}: {e}",
            category=UiPathErrorCategory.USER,
        ) from e

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _is_blocked_ip(ip):
            raise _blocked_host_error(
                f"Host {host!r} resolves to non-public address {ip}."
            )


async def _validate_request_hook(request: httpx.Request) -> None:
    """httpx request hook that runs the SSRF guard on every hop."""
    await _assert_public_url(str(request.url))


def create_http_request_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the http-request internal tool from resource configuration.

    ``llm`` is accepted for signature parity with the other internal tool
    factories but is unused: the tool issues a plain HTTP request.
    """
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_output_model(HTTP_REQUEST_OUTPUT_SCHEMA, resource.name)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for internal tools
    )
    async def http_request_tool_fn(**kwargs: Any) -> dict[str, Any]:
        url = kwargs.get("url")
        if not url:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.INVALID_INPUT_ARGUMENT,
                title="Missing required argument",
                detail="Argument 'url' is required.",
                category=UiPathErrorCategory.USER,
            )
        # Default to https when no scheme is given (e.g. "google.com").
        url = _normalize_url(url)

        method = (kwargs.get("method") or "GET").upper()
        if method not in HTTP_REQUEST_METHODS:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.INVALID_INPUT_ARGUMENT,
                title="Unsupported HTTP method",
                detail=(
                    f"Unsupported HTTP method {method!r}; expected one of "
                    f"{', '.join(HTTP_REQUEST_METHODS)}."
                ),
                category=UiPathErrorCategory.USER,
            )

        headers = kwargs.get("headers") or None
        params = kwargs.get("params") or None

        timeout = kwargs.get("timeout")
        if timeout is None:
            timeout = HTTP_REQUEST_DEFAULT_TIMEOUT_SECONDS

        # A dict/list body is sent as JSON; anything else is sent as raw content.
        request_kwargs: dict[str, Any] = {}
        body = kwargs.get("body")
        if body is not None:
            if isinstance(body, (dict, list)):
                request_kwargs["json"] = body
            elif isinstance(body, (str, bytes)):
                request_kwargs["content"] = body
            else:
                request_kwargs["content"] = str(body)

        # Build from get_httpx_client_kwargs() (enforced SSL/proxy config), but
        # drop the merged UiPath platform headers so internal licensing context
        # is never leaked to an arbitrary third-party host.
        client_kwargs = get_httpx_client_kwargs()
        client_kwargs.pop("headers", None)

        try:
            async with httpx.AsyncClient(
                event_hooks={"request": [_validate_request_hook]},
                max_redirects=_MAX_REDIRECTS,
                **client_kwargs,
            ) as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                    **request_kwargs,
                )
        except AgentRuntimeError:
            raise
        except httpx.TimeoutException as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.HTTP_ERROR,
                title="HTTP request timed out",
                detail=f"Request to {url!r} timed out after {timeout}s: {e}",
                category=UiPathErrorCategory.USER,
            ) from e
        except httpx.HTTPError as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.HTTP_ERROR,
                title="HTTP request failed",
                detail=f"Request to {url!r} failed: {e}",
                category=UiPathErrorCategory.USER,
            ) from e

        # Non-2xx responses are returned to the agent rather than raised, so it
        # can react to the status code.
        return {
            "statusCode": response.status_code,
            "headers": {k: v for k, v in response.headers.items()},
            "body": response.text,
        }

    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=http_request_tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(awrapper=job_attachment_wrapper)
    return tool
