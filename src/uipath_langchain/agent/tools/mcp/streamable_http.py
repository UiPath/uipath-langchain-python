"""Session-aware adapter for the MCP SDK's Streamable HTTP transport.

The MCP SDK owns the transport implementation. UiPath only adds asynchronous,
externally-persistable session ID storage through :class:`SessionInfo`, and
describes how that ID travels on the wire through :class:`SessionIdentityWire`.

The two eras identify a connection differently. Legacy servers mint an
``mcp-session-id`` and return it on the initialize response. ``2026-07-28`` has
no session identity at all, so UiPath mints its own ID and sends it on that same
header purely as a routing key -- a modern server ignores it, and the gateway
keeps routing on the header it already knows. Both are the same operation from
the transport's point of view -- read an ID, put it on the request, maybe read one
back -- so the transport stays ignorant of MCP negotiation and takes the
difference as data.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx2
from mcp.client.streamable_http import (
    streamable_http_client as sdk_streamable_http_client,
)
from uipath._utils._ssl_context import get_httpx_client_kwargs

MCP_SESSION_ID = "mcp-session-id"
MCP_PROTOCOL_VERSION = "mcp-protocol-version"

#: Every header this adapter manages. Any not selected by the active wire is
#: removed from the request, so a narrowing wire cannot leave a stale header on.
_IDENTITY_HEADERS = (MCP_SESSION_ID,)

logger = logging.getLogger(__name__)


class SessionInfo:
    """Store the MCP session ID and allow subclasses to persist it externally."""

    #: Class-level default so a subclass that skips ``super().__init__()`` still
    #: reads as "version not known" rather than raising.
    protocol_version: str | None = None

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        # The version the stored session was negotiated at. Held alongside the
        # ID because it cannot be recovered from the wire: responses carry only
        # the session ID. A subclass that persists the ID externally should
        # persist this too, so a later run can resume the session without
        # re-running the handshake to learn its version.
        self.protocol_version: str | None = None

    async def get_session_id(self) -> str | None:
        """Return the current session ID, or ``None`` when no session exists."""
        return self.session_id

    async def set_session_id(self, session_id: str | None) -> None:
        """Store a server-assigned session ID, or clear it with ``None``."""
        self.session_id = session_id

    async def get_protocol_version(self) -> str | None:
        """Return the version the stored session was negotiated at, if known.

        ``None`` means "not known", not "no session": a store written by an
        older revision holds an ID and no version, and a resumed connection
        then has to learn the version from the server.
        """
        return self.protocol_version

    async def set_protocol_version(self, protocol_version: str | None) -> None:
        """Record the negotiated version, or clear it with ``None``."""
        self.protocol_version = protocol_version


@dataclass(frozen=True)
class SessionIdentityWire:
    """How a stored session ID travels on the wire for one protocol era."""

    request_headers: tuple[str, ...] = (MCP_SESSION_ID,)
    """Headers the stored ID is sent on. Empty sends none."""

    capture_response_header: str | None = MCP_SESSION_ID
    """Header a server-assigned ID is read from, or ``None`` when the client mints it.

    ``None`` also protects a client-minted ID: nothing on the wire can overwrite
    the routing key mid-connection.
    """


#: Legacy handshake: the server mints the ID and returns it on a response header.
#: Also the starting wire for ``auto``, whose era is unknown when the transport
#: opens -- safe either way, because a modern server simply never sends the header
#: back.
LEGACY_IDENTITY = SessionIdentityWire()

#: Modern era: the same request header, but the client mints the value and no
#: server-assigned ID is ever read back.
MODERN_IDENTITY = SessionIdentityWire(
    request_headers=(MCP_SESSION_ID,),
    capture_response_header=None,
)


@dataclass
class SessionIdentity:
    """Mutable holder the transport reads on every request.

    The transport is opened before negotiation runs, so an era-specific wire
    cannot be fixed at construction time. A strategy narrows ``wire`` once it
    knows the era, and in-flight requests pick that up on their next call.
    """

    wire: SessionIdentityWire = field(default_factory=lambda: LEGACY_IDENTITY)


@asynccontextmanager
async def streamable_http_client(
    url: str,
    *,
    http_client: httpx2.AsyncClient | None = None,
    terminate_on_close: bool = True,
    session_info: SessionInfo | None = None,
    identity: SessionIdentity | None = None,
) -> AsyncGenerator[tuple[Any, Any], None]:
    """Open the SDK transport while synchronizing its session header externally.

    MCP 2 removed the transport's ``get_session_id`` callback. Request and
    response hooks preserve UiPath's persisted-session behavior without
    maintaining a private copy of the SDK transport.

    Args:
        url: The MCP server endpoint.
        http_client: An authenticated client to reuse. One is created and owned
            when omitted.
        terminate_on_close: Send ``DELETE`` for the session on exit. Already a
            no-op in the modern era, which has no session to terminate.
        session_info: Store for the session ID. A plain in-memory one is used
            when omitted.
        identity: How the ID travels on the wire. Defaults to the legacy
            handshake behavior.
    """
    info = session_info or SessionInfo()
    session_identity = identity or SessionIdentity()
    owns_client = http_client is None
    client = http_client or httpx2.AsyncClient(
        # A caller that supplies no client still gets the repo's SSL and proxy
        # configuration; only the MCP read timeout is layered on top, since SSE
        # streams outlive a default one.
        **{
            **get_httpx_client_kwargs(),
            "timeout": httpx2.Timeout(30, read=300),
        }
    )
    restored_session_id = await info.get_session_id()
    sdk_session_id: str | None = None
    session_persistence_lock = asyncio.Lock()

    async def apply_session_id(request: httpx2.Request) -> None:
        wire = session_identity.wire
        session_id = await info.get_session_id()
        for header in _IDENTITY_HEADERS:
            if session_id is not None and header in wire.request_headers:
                request.headers[header] = session_id
            else:
                request.headers.pop(header, None)

    async def capture_session_id(response: httpx2.Response) -> None:
        nonlocal sdk_session_id
        capture_header = session_identity.wire.capture_response_header
        if capture_header is None:
            return
        session_id = response.headers.get(capture_header)
        if session_id is None:
            return
        try:
            request_body = json.loads(response.request.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            request_body = None
        if not (
            isinstance(request_body, dict)
            and request_body.get("method") == "initialize"
        ):
            # Only the handshake assigns a session, which is how the SDK's own
            # transport reads it too. Persisting from any response would let a
            # proxy echoing the header replace a client-minted routing key --
            # reachable in ``auto`` mode, whose probe runs on the legacy wire
            # before the era is known.
            return
        sdk_session_id = session_id
        async with session_persistence_lock:
            if await info.get_session_id() != session_id:
                await info.set_session_id(session_id)

    async def terminate_restored_session() -> None:
        if session_identity.wire.capture_response_header is None:
            # This era's ID is minted by the client for routing, not assigned by
            # the server, so there is no server-side session to terminate.
            # Deleting it would tear down a live connection on a restored run.
            return
        current_session_id = await info.get_session_id()
        if (
            not terminate_on_close
            or restored_session_id is None
            or current_session_id != restored_session_id
            or current_session_id == sdk_session_id
        ):
            return
        try:
            await client.delete(url)
        except Exception as error:  # pragma: no cover - best-effort cleanup
            logger.warning("Persisted MCP session termination failed: %s", error)

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
                    try:
                        yield streams
                    finally:
                        await terminate_restored_session()
        else:
            async with sdk_streamable_http_client(
                url,
                http_client=client,
                terminate_on_close=terminate_on_close,
            ) as streams:
                try:
                    yield streams
                finally:
                    await terminate_restored_session()
    finally:
        client.event_hooks["request"].remove(apply_session_id)
        client.event_hooks["response"].remove(capture_session_id)
