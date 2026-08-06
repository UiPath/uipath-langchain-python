"""Session-aware adapter for the MCP SDK's Streamable HTTP transport.

The MCP SDK owns the transport implementation. UiPath only adds asynchronous,
externally-persistable session ID storage through :class:`SessionInfo`.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from mcp.client.streamable_http import (
    streamable_http_client as sdk_streamable_http_client,
)

MCP_SESSION_ID = "mcp-session-id"


class SessionInfo:
    """Store the MCP session ID and allow subclasses to persist it externally."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id

    async def get_session_id(self) -> str | None:
        """Return the current session ID, or ``None`` when no session exists."""
        return self.session_id

    async def set_session_id(self, session_id: str | None) -> None:
        """Store a server-assigned session ID, or clear it with ``None``."""
        self.session_id = session_id


@asynccontextmanager
async def streamable_http_client(
    url: str,
    *,
    http_client: httpx2.AsyncClient | None = None,
    terminate_on_close: bool = True,
    session_info: SessionInfo | None = None,
) -> AsyncGenerator[tuple[Any, Any], None]:
    """Open the SDK transport while synchronizing its session header externally.

    MCP 2 removed the transport's ``get_session_id`` callback. Request and
    response hooks preserve UiPath's persisted-session behavior without
    maintaining a private copy of the SDK transport.
    """
    info = session_info or SessionInfo()
    owns_client = http_client is None
    client = http_client or httpx2.AsyncClient(
        follow_redirects=True,
        timeout=httpx2.Timeout(30, read=300),
    )

    async def apply_session_id(request: httpx2.Request) -> None:
        session_id = await info.get_session_id()
        if session_id is None:
            request.headers.pop(MCP_SESSION_ID, None)
        else:
            request.headers[MCP_SESSION_ID] = session_id

    async def capture_session_id(response: httpx2.Response) -> None:
        session_id = response.headers.get(MCP_SESSION_ID)
        if session_id is not None:
            await info.set_session_id(session_id)

    client.event_hooks["request"].append(apply_session_id)
    client.event_hooks["response"].append(capture_session_id)
    try:
        if owns_client:
            async with client:
                async with sdk_streamable_http_client(
                    url,
                    http_client=client,
                    terminate_on_close=terminate_on_close,
                ) as streams:
                    yield streams
        else:
            async with sdk_streamable_http_client(
                url,
                http_client=client,
                terminate_on_close=terminate_on_close,
            ) as streams:
                yield streams
    finally:
        client.event_hooks["request"].remove(apply_session_id)
        client.event_hooks["response"].remove(capture_session_id)
