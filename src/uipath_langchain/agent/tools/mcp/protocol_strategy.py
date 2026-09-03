"""Era-specific session lifecycle for one MCP connection.

MCP has two negotiation eras. The legacy handshake (``2024-11-05`` through
``2025-11-25``) agrees a protocol version through ``initialize`` and identifies
the connection with a server-minted ``mcp-session-id``. The modern era
(``2026-07-28``) replaces the handshake with a stateless ``server/discover``
probe and has no session identity at all: every request re-declares the
protocol version, client info and capabilities.

Negotiation itself is one call in either era. What genuinely differs is the
*session lifecycle* -- how a connection is negotiated, whether a restored one can
be reused, which errors a reconnect could fix, and how the connection is
identified on the wire. Those four concerns are what :class:`ProtocolStrategy`
abstracts.
"""

import logging
from typing import Literal, Protocol, runtime_checkable
from uuid import uuid4

from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CONNECTION_CLOSED,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    UNSUPPORTED_PROTOCOL_VERSION,
    DiscoverResult,
    UnsupportedProtocolVersionErrorData,
)
from mcp.types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    LATEST_MODERN_VERSION,
    MODERN_PROTOCOL_VERSIONS,
)
from pydantic import ValidationError

from .streamable_http import (
    LEGACY_IDENTITY,
    MODERN_IDENTITY,
    SessionIdentity,
    SessionInfo,
)

logger = logging.getLogger(__name__)

ProtocolMode = Literal["legacy", "auto", "modern"]

#: Session-disconnect markers that make an ``INVALID_REQUEST`` recoverable.
_SESSION_LOST_MARKERS = ("terminated", "expired", "invalid", "not found")


def is_session_error(error: MCPError) -> bool:
    """Check whether an ``MCPError`` reports a lost legacy session.

    Args:
        error: The error to classify.

    Returns:
        True when the error indicates the session is gone rather than the
        request being wrong.
    """
    if error.code == CONNECTION_CLOSED:
        return True
    message = error.message.lower()
    return (
        error.code in (32600, INVALID_REQUEST)
        and "session" in message
        and any(marker in message for marker in _SESSION_LOST_MARKERS)
    )


@runtime_checkable
class ProtocolStrategy(Protocol):
    """Negotiation and recovery policy for one protocol era."""

    identity: SessionIdentity
    """How this era identifies the connection on the wire."""

    async def connect(self, session: ClientSession, info: SessionInfo) -> None:
        """Bring ``session`` to a negotiated state, reusing persisted state if any."""
        ...

    def is_recoverable(self, error: MCPError, restored_id: str | None) -> bool:
        """Report whether opening a fresh connection could plausibly fix ``error``."""
        ...

    async def reset(self, info: SessionInfo) -> None:
        """Discard the persisted state that made the last connection fail."""
        ...


class LegacyHandshakeStrategy:
    """Negotiate through ``initialize`` and reuse server-minted sessions.

    A restored session is re-negotiated rather than probed. The server routes
    requests purely by the ``mcp-session-id`` header and creates a new session
    only when no header is present, so an ``initialize`` carrying a restored ID
    lands *inside* that session and returns the version it was negotiated at.
    """

    def __init__(self) -> None:
        self.identity = SessionIdentity(LEGACY_IDENTITY)

    async def connect(self, session: ClientSession, info: SessionInfo) -> None:
        """Run the handshake, resuming a persisted session when one exists."""
        restored_id = await info.get_session_id()
        if restored_id is None:
            await session.initialize()
            logger.info(
                "MCP session initialized with session ID: %s",
                await info.get_session_id(),
            )
            return

        try:
            result = await session.initialize()
        except MCPError as error:
            if error.code == CONNECTION_CLOSED:
                # The transport died, which says nothing about the session.
                # Clearing the ID here would destroy an externally persisted
                # session -- permanently, for a store-backed SessionInfo -- over a
                # transient failure. Let recovery reopen and resume instead.
                raise
            # The persisted session is gone, or this server refuses a second
            # handshake. Either way a clean session is the correct fallback; the
            # transport survives a rejected request, so the same one is reused.
            logger.info(
                "Persisted MCP session %s was rejected (%s); starting a new session",
                restored_id,
                error.code,
            )
            await self.reset(info)
            await session.initialize()
            logger.info(
                "MCP session initialized with session ID: %s",
                await info.get_session_id(),
            )
            return

        current_id = await info.get_session_id()
        if current_id == restored_id:
            logger.info(
                "Reusing externally persisted MCP session %s at %s",
                restored_id,
                result.protocol_version,
            )
        else:
            # A server that ignores the session header mints a replacement. The
            # persisted session is lost, but the connection is usable.
            logger.info(
                "Server replaced persisted MCP session %s with %s at %s",
                restored_id,
                current_id,
                result.protocol_version,
            )

    def is_recoverable(self, error: MCPError, restored_id: str | None) -> bool:
        """Recognize explicit and restored-session disconnect responses.

        The SDK transport only knows session IDs received during its own
        lifetime. When UiPath restores an externally persisted ID, the request
        hook supplies it but the transport maps a bare HTTP 404 to
        ``METHOD_NOT_FOUND``. With a persisted ID on that request, Streamable
        HTTP defines the 404 as an invalid session, so a fresh handshake is safe.

        ``restored_id`` is the ID currently stored, not specifically the one
        restored when the connection opened. The distinction does not matter in
        practice: the SDK only produces this exact shape for a session it does
        not know about.
        """
        if is_session_error(error):
            return True
        if error.code != METHOD_NOT_FOUND or error.message != "Not Found":
            return False
        return restored_id is not None

    async def reset(self, info: SessionInfo) -> None:
        """Clear the stale session ID so the next handshake starts clean."""
        await info.set_session_id(None)


class ModernDiscoveryStrategy:
    """Negotiate through ``server/discover`` and carry a UiPath affinity ID.

    ``2026-07-28`` has no session identity, so there is nothing to resume and no
    session-loss error to recover from. UiPath still needs to reach the same warm
    serverless instance across requests and runs, so this strategy mints its own
    ID and sends it on ``mcp-session-id`` -- purely as a routing key, since a
    modern server has no session to attach it to and ignores it. Reusing that
    header rather than inventing one means the gateway needs no change: it keeps
    routing on the header it already routes on today.

    Unlike a server-assigned session, the ID is available on the very first
    request -- ``server/discover`` included -- so even the cold start is
    attributable.
    """

    def __init__(self) -> None:
        self.identity = SessionIdentity(MODERN_IDENTITY)

    async def connect(self, session: ClientSession, info: SessionInfo) -> None:
        """Mint an affinity ID if needed, then probe ``server/discover``."""
        await mint_affinity_id(info)
        result = await session.discover()
        logger.info(
            "MCP modern discovery negotiated %s (server supports %s)",
            session.protocol_version,
            list(result.supported_versions),
        )

    def is_recoverable(self, error: MCPError, restored_id: str | None) -> bool:
        """Retry only a dropped connection.

        Every modern request is self-contained, so no server-side session can be
        lost. Retrying anything but a transport failure spends the retry budget
        on an error a reconnect cannot fix.
        """
        return error.code == CONNECTION_CLOSED

    async def reset(self, info: SessionInfo) -> None:
        """Keep the affinity ID so a reconnect returns to the same instance."""
        return


class AutoStrategy:
    """Probe for the modern era, falling back to the legacy handshake.

    The affinity ID is minted *before* the probe, so ``server/discover`` reaches
    the same instance the tool calls will: on a serverless gateway, an unpinned
    probe would warm one instance and the first call would land on another. A
    legacy server sees an ID it never issued; it is cleared again before the
    handshake so that server is not asked to resume a session that never was.

    The era is re-resolved on every ``connect``, so a server upgraded mid-run is
    handled.
    """

    def __init__(self) -> None:
        # Both eras send the ID on the same header, so the transport can open on
        # the legacy wire before the era is known: a modern server simply never
        # sends one back, leaving nothing to capture.
        self.identity = SessionIdentity(LEGACY_IDENTITY)
        self._legacy = LegacyHandshakeStrategy()
        self._modern = ModernDiscoveryStrategy()
        self._resolved: ProtocolStrategy = self._legacy

    async def connect(self, session: ClientSession, info: SessionInfo) -> None:
        """Negotiate an era, then narrow this strategy to it."""
        self.identity.wire = LEGACY_IDENTITY
        # Widen back to the conservative era first: if the probe raises, the
        # previous connection's resolution must not decide how this failure is
        # recovered from.
        self._resolved = self._legacy
        restored_id = await info.get_session_id()
        if restored_id is None:
            await mint_affinity_id(info)

        if await probe_modern_era(session):
            self._resolved = self._modern
            logger.info("MCP era resolved to modern (%s)", session.protocol_version)
        else:
            if restored_id is None:
                # The ID was minted for this connection and never named a
                # session on this server. Sending it into the handshake would
                # only earn a rejection; a restored ID, by contrast, may well
                # be a live legacy session and is left for the handshake.
                await info.set_session_id(None)
            await self._legacy.connect(session, info)
            self._resolved = self._legacy
            logger.info("MCP era resolved to legacy (%s)", session.protocol_version)
        self.identity.wire = self._resolved.identity.wire

    def is_recoverable(self, error: MCPError, restored_id: str | None) -> bool:
        """Apply the resolved era's recovery policy."""
        return self._resolved.is_recoverable(error, restored_id)

    async def reset(self, info: SessionInfo) -> None:
        """Apply the resolved era's reset policy."""
        await self._resolved.reset(info)


async def probe_modern_era(session: ClientSession) -> bool:
    """Probe ``server/discover`` and adopt the modern era if the peer speaks it.

    Only positive evidence of a modern server counts; anything else reports
    ``False`` so the caller can fall back to the ``initialize`` handshake. That
    is a denylist, not an allowlist: every JSON-RPC error falls back, including
    the HTTP-layer 4xx the transport synthesizes into one, as does a result
    that fails to parse or advertises no modern version. A ``-32022`` naming a
    mutual modern version earns one re-probe at that version.

    Built on the session's public ``send_discover`` / ``adopt`` seam rather than
    the SDK's private ``mode="auto"`` helper, so an SDK patch cannot move this
    policy out from under the client.

    Args:
        session: The un-negotiated session to probe.

    Returns:
        True when ``server/discover`` succeeded and was adopted.

    Raises:
        MCPError: The server is modern-only yet shares no version with this
            client -- a ``-32022`` whose ``supported`` list has no handshake
            version -- so no era can work.
    """
    version = LATEST_MODERN_VERSION
    for attempt in range(2):
        try:
            raw = await session.send_discover(version)
        except MCPError as error:
            supported = _versions_supported_by(error)
            if supported is None:
                return False
            mutual = [v for v in MODERN_PROTOCOL_VERSIONS if v in supported]
            if mutual and attempt == 0:
                version = mutual[-1]
                continue
            if not any(v in HANDSHAKE_PROTOCOL_VERSIONS for v in supported):
                raise
            return False
        try:
            result = DiscoverResult.model_validate(raw)
        except ValidationError:
            return False
        if not any(v in result.supported_versions for v in MODERN_PROTOCOL_VERSIONS):
            # A discover-answering server advertising only handshake versions is
            # a legacy advertisement, not an incompatibility.
            return False
        session.adopt(result)
        return True
    return False


def _versions_supported_by(error: MCPError) -> list[str] | None:
    """Read the ``supported`` list off a ``-32022`` error, or ``None``."""
    if error.code != UNSUPPORTED_PROTOCOL_VERSION:
        return None
    try:
        data = UnsupportedProtocolVersionErrorData.model_validate(error.error.data)
    except ValidationError:
        return None
    return data.supported


async def mint_affinity_id(info: SessionInfo) -> str:
    """Ensure ``info`` holds a client-minted ID, and return it.

    Stored through the same accessors as a server-assigned session ID, so an
    external ``SessionInfo`` implementation persists it without modification and
    a later run resumes on the same instance.
    """
    session_id = await info.get_session_id()
    if session_id is None:
        session_id = uuid4().hex
        await info.set_session_id(session_id)
        logger.info("Minted UiPath MCP affinity ID %s", session_id)
    return session_id


def build_protocol_strategy(mode: ProtocolMode) -> ProtocolStrategy:
    """Create the strategy for a protocol mode.

    Args:
        mode: ``"legacy"`` for the ``initialize`` handshake only, ``"modern"``
            for ``server/discover`` only, or ``"auto"`` to probe for the modern
            era and fall back to the handshake.

    Returns:
        The strategy implementing that mode.

    Raises:
        ValueError: ``mode`` is not a known protocol mode.
    """
    if mode == "legacy":
        return LegacyHandshakeStrategy()
    if mode == "modern":
        return ModernDiscoveryStrategy()
    if mode == "auto":
        return AutoStrategy()
    raise ValueError(
        f"Unknown MCP protocol mode {mode!r}; expected 'legacy', 'auto' or 'modern'"
    )
