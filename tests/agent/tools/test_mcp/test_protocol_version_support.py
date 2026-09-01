"""Guards on the SDK protocol-version constraints our client path depends on.

These assert facts about the MCP SDK rather than about UiPath code, which is
unusual for a unit test. They earn their place because each one pins an external
constraint that dictates how the MCP integration is built and tested, and each
tells you what to change when it flips. They are deliberately cheap: no sockets,
no servers.
"""

import inspect

import pytest
from mcp import Client, ClientSession
from mcp.types.version import (
    HANDSHAKE_PROTOCOL_VERSIONS,
    LATEST_HANDSHAKE_VERSION,
    MODERN_PROTOCOL_VERSIONS,
)

from uipath_langchain.agent.tools.mcp.mcp_client import (
    LEGACY_STREAMABLE_HTTP_VERSIONS,
)

# The version the 2025-06-18 testcase leg would need a client to request.
HANDSHAKE_PIN = "2025-06-18"


def test_low_level_session_cannot_choose_a_protocol_version() -> None:
    """``ClientSession.initialize()`` takes no version argument.

    ``McpClient`` builds on the low-level session, so the version it offers is
    not ours to pick -- it is always ``LATEST_HANDSHAKE_VERSION`` and the server
    chooses from there. If ``initialize`` ever accepts a version, the
    ``simple-http-mcp`` testcase can drive real servers at older versions
    instead of simulated ones.
    """
    parameters = inspect.signature(ClientSession.initialize).parameters
    assert list(parameters) == ["self"], (
        "ClientSession.initialize() gained parameters; the MCP integration may "
        "now be able to request a specific protocol version"
    )


def test_sdk_client_rejects_a_handshake_era_mode_pin() -> None:
    """The high-level ``Client`` accepts only modern versions in ``mode=``.

    This is why ``testcases/simple-http-mcp`` covers ``2025-06-18`` with a
    simulated endpoint. A real server *will* negotiate that version when the
    ``initialize`` params ask for it, so the limitation is purely client-side.
    When this stops raising, make that leg use a real server.
    """
    with pytest.raises(ValueError):
        Client("http://127.0.0.1/mcp", mode=HANDSHAKE_PIN)


def test_modern_versions_are_outside_the_handshake_set() -> None:
    """Modern versions are unreachable through the ``initialize`` handshake.

    ``McpClient`` only ever sends ``initialize``, so a server that speaks only a
    modern version cannot be reached. When a modern version joins the handshake
    set, that limitation is gone and the testcase's modern-only negative should
    be inverted.
    """
    assert LATEST_HANDSHAKE_VERSION in HANDSHAKE_PROTOCOL_VERSIONS
    overlap = set(MODERN_PROTOCOL_VERSIONS) & set(HANDSHAKE_PROTOCOL_VERSIONS)
    assert not overlap, (
        f"Modern version(s) {sorted(overlap)} entered the handshake set; the "
        "low-level client path may now reach the modern era"
    )


def test_restored_sessions_cover_every_handshake_version_but_the_oldest() -> None:
    """``McpClient`` adopts restored sessions on a narrower set than it connects on.

    A fresh connection accepts any handshake version, but restored-session
    adoption skips ``2024-11-05``, which predates Streamable HTTP sessions. This
    pins that intentional gap so it is not widened by accident when the SDK adds
    or removes a handshake version.
    """
    expected = tuple(v for v in HANDSHAKE_PROTOCOL_VERSIONS if v >= "2025-03-26")
    assert LEGACY_STREAMABLE_HTTP_VERSIONS == expected
    assert "2024-11-05" not in LEGACY_STREAMABLE_HTTP_VERSIONS
