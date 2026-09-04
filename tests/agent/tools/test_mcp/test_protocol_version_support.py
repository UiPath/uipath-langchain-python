"""Guards on the SDK facts the MCP protocol strategies are built on.

These assert facts about the MCP SDK rather than about UiPath code, which is
unusual for a unit test. They earn their place because each one pins an external
constraint that dictates how the MCP integration is built, and each says what to
change when it flips. They are deliberately cheap: no sockets, no servers.
"""

import inspect

from mcp import ClientSession
from mcp.types import UnsupportedProtocolVersionErrorData
from mcp.types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    LATEST_HANDSHAKE_VERSION,
    LATEST_MODERN_VERSION,
    MODERN_PROTOCOL_VERSIONS,
)


def test_the_auto_probe_builds_on_public_session_methods() -> None:
    """``probe_modern_era`` owns the ``auto`` policy on public ``ClientSession`` seams.

    It sends the probe through ``send_discover(version)`` and installs the
    result through ``adopt(result)`` -- the same two calls the SDK's private
    ``mode="auto"`` helper is built from, which is deliberately not imported.
    A ``-32022`` is read through ``UnsupportedProtocolVersionErrorData.supported``.
    If any of these change shape, the probe needs the matching change.
    """
    assert list(inspect.signature(ClientSession.send_discover).parameters) == [
        "self",
        "version",
    ], "ClientSession.send_discover() changed shape; update probe_modern_era"
    assert list(inspect.signature(ClientSession.adopt).parameters) == [
        "self",
        "result",
    ], "ClientSession.adopt() changed shape; update probe_modern_era"
    assert "supported" in UnsupportedProtocolVersionErrorData.model_fields, (
        "-32022 error data lost its 'supported' list; probe_modern_era can no "
        "longer tell a modern-only server from a legacy one"
    )


def test_the_low_level_session_reaches_the_modern_era() -> None:
    """``ClientSession.discover()`` is why ``McpClient`` needs no high-level client.

    ``discover`` takes no version argument: it always proposes
    ``LATEST_MODERN_VERSION`` and the SDK owns the ``-32022`` retry at a mutual
    version. If it gains parameters, ``ModernDiscoveryStrategy`` may be able to
    pin a version explicitly.
    """
    assert hasattr(ClientSession, "discover"), (
        "ClientSession lost discover(); the modern era is no longer reachable "
        "from the low-level session and ModernDiscoveryStrategy needs rework"
    )
    parameters = inspect.signature(ClientSession.discover).parameters
    assert list(parameters) == ["self"], (
        "ClientSession.discover() gained parameters; the modern strategy may now "
        "be able to request a specific protocol version"
    )
    assert LATEST_MODERN_VERSION in MODERN_PROTOCOL_VERSIONS


def test_initialize_cannot_choose_a_protocol_version() -> None:
    """``ClientSession.initialize()`` takes no version argument.

    The legacy strategy always offers ``LATEST_HANDSHAKE_VERSION`` and the server
    counters with what it supports, which is why a resumed session re-runs the
    handshake to learn its version rather than guessing at one.
    """
    parameters = inspect.signature(ClientSession.initialize).parameters
    assert list(parameters) == ["self"], (
        "ClientSession.initialize() gained parameters; the legacy strategy may "
        "now be able to request a specific protocol version"
    )
    assert LATEST_HANDSHAKE_VERSION in HANDSHAKE_PROTOCOL_VERSIONS


def test_the_two_eras_share_no_protocol_version() -> None:
    """Disjoint version sets are why there are two strategies rather than one.

    A version reachable through both ``initialize`` and ``server/discover`` would
    make the era a property of the server rather than of the negotiation, and
    ``AutoStrategy`` could stop probing.
    """
    overlap = set(MODERN_PROTOCOL_VERSIONS) & set(HANDSHAKE_PROTOCOL_VERSIONS)
    assert not overlap, (
        f"Version(s) {sorted(overlap)} are in both eras; the handshake may now "
        "reach the modern protocol and AutoStrategy's probe may be redundant"
    )
