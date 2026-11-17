"""Tests for redirect handling in the streamable HTTP transport."""

import json
from collections.abc import Callable

import anyio
import httpx
from mcp.shared.message import ClientMessageMetadata, SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCRequest, JSONRPCResponse
from uipath._utils._ssl_context import get_httpx_client_kwargs

from uipath_langchain.agent.tools.mcp.streamable_http import (
    MAX_RECONNECTION_ATTEMPTS,
    streamable_http_client,
)

ENDPOINT = "https://mcp.example/mcp"
MOVED = "https://mcp.example/mcp/"
OTHER_ORIGIN = "https://other.example/mcp"

Handler = Callable[[httpx.Request], httpx.Response]


def _redirect(location: str) -> httpx.Response:
    return httpx.Response(307, headers={"location": location})


def _sse(*events: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(events).encode(),
    )


def _result_event(request_id: int, event_id: str) -> str:
    payload = {"jsonrpc": "2.0", "id": request_id, "result": {"tools": []}}
    return f"event: message\nid: {event_id}\ndata: {json.dumps(payload)}\n\n"


def _priming_event(event_id: str) -> str:
    return f"event: message\nid: {event_id}\nretry: 0\ndata: \n\n"


def _flatten(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for sub in error.exceptions for leaf in _flatten(sub)]
    return [error]


class _Recorder:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def to(self, url: str) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url) == url]


async def _send_request(
    recorder: _Recorder,
    resumption_token: str | None = None,
    wait_for: Callable[[], bool] | None = None,
) -> tuple[SessionMessage | Exception | None, list[BaseException]]:
    message = SessionMessage(
        JSONRPCMessage(JSONRPCRequest(jsonrpc="2.0", id=7, method="tools/list")),
        metadata=ClientMessageMetadata(resumption_token=resumption_token)
        if resumption_token
        else None,
    )
    client = httpx.AsyncClient(
        **get_httpx_client_kwargs(
            headers={"Authorization": "Bearer test-secret-token"}
        ),
        transport=httpx.MockTransport(recorder),
    )
    received: SessionMessage | Exception | None = None
    errors: list[BaseException] = []
    try:
        with anyio.fail_after(10):
            async with client:
                async with streamable_http_client(
                    ENDPOINT, http_client=client, terminate_on_close=False
                ) as (read_stream, write_stream):
                    await write_stream.send(message)
                    if wait_for is None:
                        received = await read_stream.receive()
                    else:
                        while not wait_for():
                            await anyio.sleep(0.01)
    except BaseException as exc:
        errors = _flatten(exc)
    return received, errors


def _response_id(received: SessionMessage | Exception | None) -> object:
    assert isinstance(received, SessionMessage)
    assert isinstance(received.message.root, JSONRPCResponse)
    return received.message.root.id


class TestStreamableHttpRedirects:
    async def test_https_to_http_redirect_is_not_followed(self) -> None:
        recorder = _Recorder(lambda request: _redirect("http://mcp.example/mcp"))

        _, errors = await _send_request(recorder)

        status_errors = [e for e in errors if isinstance(e, httpx.HTTPStatusError)]
        assert len(status_errors) == 1
        assert status_errors[0].response.status_code == 307
        assert "downgrade" in str(status_errors[0])
        assert [str(r.url) for r in recorder.requests] == [ENDPOINT]

    async def test_resumption_follows_same_origin_redirect(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == ENDPOINT:
                return _redirect(MOVED)
            return _sse(_result_event(7, "evt-2"))

        recorder = _Recorder(handler)

        received, errors = await _send_request(recorder, resumption_token="evt-1")

        assert errors == []
        assert _response_id(received) == 7
        moved = recorder.to(MOVED)
        assert [r.method for r in moved] == ["GET"]
        assert moved[0].headers["last-event-id"] == "evt-1"
        assert moved[0].headers["authorization"] == "Bearer test-secret-token"

    async def test_resumption_cross_origin_redirect_is_not_followed(self) -> None:
        recorder = _Recorder(lambda request: _redirect(OTHER_ORIGIN))

        _, errors = await _send_request(recorder, resumption_token="evt-1")

        status_errors = [e for e in errors if isinstance(e, httpx.HTTPStatusError)]
        assert len(status_errors) == 1
        assert status_errors[0].response.status_code == 307
        assert recorder.to(OTHER_ORIGIN) == []

    async def test_reconnection_follows_same_origin_redirect(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return _sse(_priming_event("evt-1"))
            if str(request.url) == ENDPOINT:
                return _redirect(MOVED)
            return _sse(_result_event(7, "evt-2"))

        recorder = _Recorder(handler)

        received, errors = await _send_request(recorder)

        assert errors == []
        assert _response_id(received) == 7
        moved = recorder.to(MOVED)
        assert [r.method for r in moved] == ["GET"]
        assert moved[0].headers["last-event-id"] == "evt-1"

    async def test_reconnection_cross_origin_redirect_is_not_followed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return _sse(_priming_event("evt-1"))
            return _redirect(OTHER_ORIGIN)

        recorder = _Recorder(handler)

        def reconnections() -> list[httpx.Request]:
            return [r for r in recorder.to(ENDPOINT) if r.method == "GET"]

        _, errors = await _send_request(
            recorder, wait_for=lambda: len(reconnections()) == MAX_RECONNECTION_ATTEMPTS
        )

        assert errors == []
        assert recorder.to(OTHER_ORIGIN) == []
        assert all(r.headers["last-event-id"] == "evt-1" for r in reconnections())
