"""Tests for A2A tool URL resolution.

Focuses on the behavior of preferring the UiPath-hosted proxy URL
(``a2a_url``) over any URL cached in the agent card.
"""

from typing import Any

import pytest
from uipath.agent.models.agent import AgentA2aResourceConfig

from uipath_langchain.agent.tools.a2a.a2a_tool import (
    _resolve_a2a_url,
    create_a2a_tools_and_clients,
)

PROXY_URL = "https://cloud.uipath.com/proxy/a2a/remote-agent"
CACHED_URL = "https://internal.example.com/agents/remote-agent"


def _make_resource(
    *,
    a2a_url: str = PROXY_URL,
    cached_agent_card: dict[str, Any] | None = None,
    is_enabled: bool = True,
) -> AgentA2aResourceConfig:
    """Build an A2A resource config for tests."""
    return AgentA2aResourceConfig(
        id="resource-id",
        name="remote-agent",
        description="A remote A2A agent",
        is_enabled=is_enabled,
        slug="remote-agent-slug",
        folder_path="Shared",
        a2a_url=a2a_url,
        cached_agent_card=cached_agent_card,
    )


def test_resolve_a2a_url_returns_proxy_url() -> None:
    resource = _make_resource(a2a_url=PROXY_URL)
    assert _resolve_a2a_url(resource) == PROXY_URL


def test_resolve_a2a_url_raises_when_missing() -> None:
    resource = _make_resource(a2a_url="")
    with pytest.raises(ValueError, match="has no URL"):
        _resolve_a2a_url(resource)


def test_create_tools_prefers_proxy_url_over_cached_card_url() -> None:
    """The proxy a2a_url must override the URL embedded in the cached card."""
    resource = _make_resource(
        a2a_url=PROXY_URL,
        cached_agent_card={
            "url": CACHED_URL,
            "name": "Remote Agent",
            "description": "cached",
            "version": "1.0.0",
            "skills": [],
            "capabilities": {},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    )

    tools, clients = create_a2a_tools_and_clients([resource])

    assert len(tools) == 1
    assert len(clients) == 1
    # The client's agent card URL is rewritten to the proxy URL.
    assert clients[0]._agent_card.url == PROXY_URL


def test_create_tools_sets_url_when_no_cached_card() -> None:
    resource = _make_resource(a2a_url=PROXY_URL, cached_agent_card=None)

    tools, clients = create_a2a_tools_and_clients([resource])

    assert len(tools) == 1
    assert clients[0]._agent_card.url == PROXY_URL


def test_create_tools_skips_disabled_resource() -> None:
    resource = _make_resource(is_enabled=False)

    tools, clients = create_a2a_tools_and_clients([resource])

    assert tools == []
    assert clients == []
