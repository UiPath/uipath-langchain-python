"""``McpClient`` driven against real MCP servers over real HTTP.

Everywhere else the client is exercised through ``httpx2.MockTransport``, which
proves the wire is shaped right but never runs a real server, a real ASGI stack,
or the gateway hop UiPath puts in front of an MCP endpoint. These tests host a
genuine ``MCPServer`` (and hand-written endpoints where a version has to be
pinned) on an ephemeral port and drive the public ``McpClient`` API against it.

Every assertion is made from the *gateway's* point of view -- what actually
arrived on the wire -- rather than from client internals, so a test fails when
the observable behaviour changes, not when the implementation moves.

See ``real_server.py`` for the harness.
"""

from typing import Any

import pytest
from mcp.shared.exceptions import MCPError

from uipath_langchain.agent.tools.mcp import SessionInfo, SessionInfoFactory

from .real_server import (
    HANDSHAKE_VERSIONS,
    LEGACY_VERSION,
    MODERN_VERSION,
    SDK_LOOKUPS,
    PinnedVersionServer,
    RecordingGateway,
    build_sdk_app,
    connected_client,
    make_client,
    make_resource_config,
    patched_sdk,
    pinned_session_factory,
    serve,
)


def negotiated_version(gateway: RecordingGateway, rpc_method: str) -> str | None:
    """Read the version the SDK stamped on the first request of one method.

    The ``mcp-protocol-version`` header is written by the SDK once negotiation
    completes, so it is the wire-visible answer to "what did we settle on".
    """
    records = gateway.for_rpc(rpc_method)
    return records[0].protocol_version if records else None


# --- negotiation per mode ---------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_mode_negotiates_the_newest_handshake_version() -> None:
    """A real SDK server answers the handshake at 2025-11-25 and mints a session."""
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="legacy") as client:
            tools = await client.list_tools()
            result = await client.call_tool("add", {"a": 2, "b": 3})
            session_id = await client.get_session_id()

    assert sorted(tool.name for tool in tools.tools) == ["add", "multiply"]
    assert result.structured_content == {"result": 5}
    assert gateway.count("initialize") == 1
    assert gateway.count("server/discover") == 0
    assert negotiated_version(gateway, "tools/call") == LEGACY_VERSION
    # The server owns the identity in this era, and it reaches the client on a
    # response header rather than being invented locally.
    assert session_id is not None
    assert gateway.server_session_ids()
    assert gateway.for_rpc("tools/call")[0].session_id == session_id


@pytest.mark.asyncio
async def test_modern_mode_negotiates_without_any_server_session() -> None:
    """2026-07-28 has no session identity, so no response may carry one."""
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            tools = await client.list_tools()
            result = await client.call_tool("add", {"a": 2, "b": 3})

    assert sorted(tool.name for tool in tools.tools) == ["add", "multiply"]
    assert result.structured_content == {"result": 5}
    assert gateway.count("server/discover") == 1
    assert gateway.count("initialize") == 0
    assert negotiated_version(gateway, "tools/call") == MODERN_VERSION
    assert gateway.server_session_ids() == [], (
        "A modern server issued a session ID; the era has no session identity "
        "and the client-minted affinity ID must be the only one in play"
    )


@pytest.mark.asyncio
async def test_auto_mode_resolves_to_modern_against_a_real_server() -> None:
    """A server answering discovery is driven as modern, handshake untouched."""
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="auto") as client:
            result = await client.call_tool("add", {"a": 4, "b": 5})

    assert result.structured_content == {"result": 9}
    assert gateway.count("server/discover") == 1
    assert gateway.count("initialize") == 0
    assert negotiated_version(gateway, "tools/call") == MODERN_VERSION
    assert gateway.server_session_ids() == []


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_legacy_against_a_handshake_only_server() -> None:
    """No discovery endpoint means the probe must fall back, honouring the offer."""
    server = PinnedVersionServer("2025-06-18")
    gateway = RecordingGateway(server.build_app())
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="auto") as client:
            result = await client.call_tool("add", {"a": 4, "b": 5})
            session_id = await client.get_session_id()

    assert result.structured_content == {"result": 9}
    assert gateway.count("server/discover") == 1
    assert server.discover_count == 0
    assert server.initialize_count == 1
    assert session_id == "session-1"
    assert negotiated_version(gateway, "tools/call") == "2025-06-18"
    # The probe is pinned like every first request, but the ID it carried was
    # minted here and never named a session on this server. The handshake must
    # not present it, or a server that routes by the header would refuse it.
    assert gateway.for_rpc("server/discover")[0].session_id is not None
    assert gateway.for_rpc("initialize")[0].session_id is None


@pytest.mark.asyncio
async def test_modern_mode_works_against_a_server_that_refuses_the_handshake() -> None:
    """A discover-only server proves modern is not silently falling back.

    This endpoint answers ``server/discover`` and rejects ``initialize``
    outright, so a client that quietly degraded to the legacy handshake could
    not complete a single call here.
    """
    server = PinnedVersionServer(MODERN_VERSION, modern_only=True)
    gateway = RecordingGateway(server.build_app())
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            result = await client.call_tool("add", {"a": 2, "b": 3})

    assert result.structured_content == {"result": 5}
    assert server.discover_count == 1
    assert server.initialize_count == 0
    assert gateway.count("initialize") == 0
    assert negotiated_version(gateway, "tools/call") == MODERN_VERSION


# --- resume across clients --------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_resume_keeps_the_originally_negotiated_version() -> None:
    """A resumed session must keep the version it was negotiated at.

    Two clients share one ``SessionInfo``, standing in for two runs of a
    playground agent whose session store outlives the process. The session ID
    surviving is only half the contract: the resumed connection must also speak
    the version that session was negotiated at. Probing candidate versions
    instead -- the pre-existing approach -- always matched the *oldest*
    handshake version, silently downgrading every later request.
    """
    gateway = RecordingGateway(build_sdk_app())
    shared = SessionInfo()
    factory = pinned_session_factory(shared)

    async with serve(gateway) as url:
        with patched_sdk(url):
            first = make_client(session_info_factory=factory, terminate_on_close=False)
            await first.call_tool("add", {"a": 1, "b": 1})
            original_session_id = await first.get_session_id()
            await first.dispose()

            resume_boundary = len(gateway.records)

            second = make_client(session_info_factory=factory, terminate_on_close=False)
            result = await second.call_tool("add", {"a": 2, "b": 2})
            resumed_session_id = await second.get_session_id()
            await second.dispose()

    assert result.structured_content == {"result": 4}
    assert original_session_id is not None
    assert resumed_session_id == original_session_id

    after_resume = gateway.records[resume_boundary:]
    assert after_resume, "The second client sent no requests"
    assert all(r.session_id == original_session_id for r in after_resume)

    # Derived from the recording, not asserted against a constant: the contract
    # is "the same version as before", so reading it back from the pre-resume
    # traffic keeps the guard honest if the server's default ever changes.
    before_resume = {
        record.protocol_version
        for record in gateway.records[:resume_boundary]
        if record.protocol_version is not None
    }
    assert len(before_resume) == 1, (
        f"The first client itself spoke {sorted(before_resume)}; the fixture no "
        "longer establishes a single negotiated version to compare against."
    )
    versions = {
        record.protocol_version
        for record in after_resume
        if record.protocol_version is not None
    }
    assert versions == before_resume, (
        f"Requests after resume negotiated {sorted(versions)} instead of "
        f"{sorted(before_resume)}. This is the silent-downgrade regression "
        "guard: a resumed session that guesses its version settles on the "
        "oldest handshake version and downgrades every later request."
    )
    resumed_calls = [r for r in after_resume if r.rpc_method == "tools/call"]
    assert resumed_calls and resumed_calls[0].protocol_version == LEGACY_VERSION
    # And it costs nothing to learn: the version was stored with the ID, so the
    # resumed connection adopts it instead of handshaking a second time.
    assert gateway.count("initialize") == 1
    assert not [r for r in after_resume if r.rpc_method == "initialize"]


@pytest.mark.asyncio
async def test_legacy_resume_survives_a_server_that_refuses_reinitialization() -> None:
    """A resumed session must not depend on being allowed to re-handshake.

    The reference TypeScript implementation answers a second ``initialize`` on a
    live session with "Server already initialized". A client that resumes by
    re-running the handshake loses the persisted session on every run against
    such a server -- and with it the gateway affinity the session ID provides.
    Adopting the stored version keeps the resume free of wire traffic, so the
    server is never asked.
    """
    server = PinnedVersionServer("2025-06-18", refuse_reinitialize=True)
    gateway = RecordingGateway(server.build_app())
    shared = SessionInfo()
    factory = pinned_session_factory(shared)

    async with serve(gateway) as url:
        with patched_sdk(url):
            first = make_client(session_info_factory=factory, terminate_on_close=False)
            await first.call_tool("add", {"a": 1, "b": 1})
            original_session_id = await first.get_session_id()
            await first.dispose()

            resume_boundary = len(gateway.records)

            second = make_client(session_info_factory=factory, terminate_on_close=False)
            result = await second.call_tool("add", {"a": 2, "b": 2})
            resumed_session_id = await second.get_session_id()
            await second.dispose()

    assert result.structured_content == {"result": 4}
    assert resumed_session_id == original_session_id == "session-1"
    # The handshake was never re-sent, so the server never had to refuse it.
    assert server.initialize_count == 1
    assert server.refused_reinitialize_count == 0
    after_resume = gateway.records[resume_boundary:]
    assert after_resume and all(
        record.session_id == original_session_id for record in after_resume
    )
    assert [
        r.protocol_version for r in after_resume if r.rpc_method == "tools/call"
    ] == ["2025-06-18"]


@pytest.mark.asyncio
async def test_unknown_persisted_session_falls_back_to_a_fresh_session() -> None:
    """A stale stored ID is rejected by the server, and the client starts clean."""
    gateway = RecordingGateway(build_sdk_app())
    stored = SessionInfo("never-existed")

    async with serve(gateway) as url:
        async with connected_client(
            url, session_info_factory=pinned_session_factory(stored)
        ) as client:
            result = await client.call_tool("add", {"a": 6, "b": 1})
            session_id = await client.get_session_id()

    assert result.structured_content == {"result": 7}
    # The refused handshake for the stale ID, then the clean one -- on the same
    # transport, because a refused request does not close the connection.
    assert gateway.count("initialize") == 2
    handshakes = gateway.for_rpc("initialize")
    assert [h.session_id for h in handshakes] == ["never-existed", None]
    assert session_id is not None and session_id != "never-existed"
    assert gateway.for_rpc("tools/call")[0].session_id == session_id


# --- affinity and disposal --------------------------------------------------


@pytest.mark.asyncio
async def test_modern_affinity_pins_one_instance_across_clients() -> None:
    """The affinity ID is what replaces gateway routing once sessions are gone.

    ``2026-07-28`` removes ``mcp-session-id`` from the protocol, so UiPath mints
    the value itself and keeps sending it on that header purely as a routing
    key. Because it is minted *before* negotiating, even ``server/discover``
    is routable -- which a server-assigned session ID never could be.
    """
    gateway = RecordingGateway(build_sdk_app())
    shared = SessionInfo()
    factory = pinned_session_factory(shared)

    async with serve(gateway) as url:
        with patched_sdk(url):
            for operands in ((1, 2), (3, 4)):
                client = make_client(
                    session_info_factory=factory, protocol_mode="modern"
                )
                try:
                    await client.call_tool("add", {"a": operands[0], "b": operands[1]})
                finally:
                    await client.dispose()

    affinity_id = await shared.get_session_id()
    assert affinity_id
    assert gateway.count("server/discover") == 2
    assert [r.session_id for r in gateway.for_rpc("server/discover")] == [
        affinity_id,
        affinity_id,
    ]
    assert gateway.unpinned() == [], (
        "Requests reached the gateway with no affinity header, so it would have "
        "had to route them blind"
    )
    assert sorted({record.instance for record in gateway.records}) == ["instance-1"]
    # A minted ID is as vulnerable as a restored one: if disposal tore it down,
    # the gateway would see a teardown for the instance it is meant to pin.
    assert gateway.http_count("DELETE") == 0


@pytest.mark.asyncio
async def test_auto_mode_pins_the_first_request() -> None:
    """In ``auto`` the probe must be pinned too, not only the calls after it.

    A serverless gateway routes on the header. An unpinned ``server/discover``
    warms one instance and the first tool call then lands on another -- the
    exact scatter the affinity ID exists to prevent.
    """
    gateway = RecordingGateway(build_sdk_app())
    shared = SessionInfo()

    async with serve(gateway) as url:
        async with connected_client(
            url,
            protocol_mode="auto",
            session_info_factory=pinned_session_factory(shared),
        ) as client:
            await client.call_tool("add", {"a": 1, "b": 2})

    affinity_id = await shared.get_session_id()
    assert affinity_id
    assert gateway.rpc_methods()[0] == "server/discover"
    assert gateway.for_rpc("server/discover")[0].session_id == affinity_id
    assert gateway.unpinned() == []
    assert sorted({record.instance for record in gateway.records}) == ["instance-1"]


@pytest.mark.asyncio
async def test_modern_disposal_does_not_delete_a_restored_affinity_id() -> None:
    """A client-minted routing key must never be torn down as if it were a session.

    A restored affinity ID looks exactly like a restored session to the
    transport. Deleting it would reach the gateway as a teardown for a live
    instance on every run after the first -- precisely the playground case.
    """
    gateway = RecordingGateway(build_sdk_app())
    restored = SessionInfo("restored-affinity")

    async with serve(gateway) as url:
        async with connected_client(
            url,
            protocol_mode="modern",
            session_info_factory=pinned_session_factory(restored),
            terminate_on_close=True,
        ) as client:
            await client.call_tool("add", {"a": 2, "b": 2})

    assert gateway.http_count("DELETE") == 0
    # A restored ID is reused rather than replaced, from the very first request.
    assert gateway.for_rpc("server/discover")[0].session_id == "restored-affinity"
    assert all(record.session_id == "restored-affinity" for record in gateway.records)
    # The ID survives disposal, so the next run returns to the same instance.
    assert await restored.get_session_id() == "restored-affinity"


@pytest.mark.asyncio
async def test_legacy_disposal_deletes_a_restored_session() -> None:
    """A restored *server* session is real state, so disposal must terminate it."""
    gateway = RecordingGateway(build_sdk_app())
    shared = SessionInfo()
    factory = pinned_session_factory(shared)

    async with serve(gateway) as url:
        with patched_sdk(url):
            first = make_client(session_info_factory=factory, terminate_on_close=False)
            await first.call_tool("add", {"a": 1, "b": 1})
            session_id = await first.get_session_id()
            await first.dispose()

            assert gateway.http_count("DELETE") == 0
            resume_boundary = len(gateway.records)

            second = make_client(session_info_factory=factory, terminate_on_close=True)
            await second.call_tool("add", {"a": 2, "b": 2})
            await second.dispose()

    deletes = [
        record
        for record in gateway.records[resume_boundary:]
        if record.http_method == "DELETE"
    ]
    assert len(deletes) == 1
    assert deletes[0].session_id == session_id


# --- retry semantics per era ------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_recovers_from_an_injected_session_termination() -> None:
    """A lost legacy session is re-established and the call retried."""
    gateway = RecordingGateway(build_sdk_app(), fault_on_tool_call=1)
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="legacy") as client:
            result = await client.call_tool("add", {"a": 3, "b": 4})

    assert result.structured_content == {"result": 7}
    assert gateway.count("initialize") == 2
    assert gateway.count("tools/call") == 2
    calls = gateway.for_rpc("tools/call")
    assert calls[0].faulted and not calls[1].faulted
    # A fresh handshake means a fresh session, so the retry cannot reuse the
    # session the gateway just declared dead.
    assert calls[0].session_id != calls[1].session_id


@pytest.mark.asyncio
async def test_modern_does_not_retry_an_injected_session_termination() -> None:
    """Reconnecting cannot restore state a self-contained request never had.

    The identical response is retried once in legacy mode. Here it must surface
    immediately instead of spending the retry budget on something a reconnect
    cannot fix.
    """
    gateway = RecordingGateway(build_sdk_app(), fault_on_tool_call=1)
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="modern") as client:
            with pytest.raises(MCPError):
                await client.call_tool("add", {"a": 3, "b": 4})

    assert gateway.count("tools/call") == 1
    assert gateway.count("server/discover") == 1
    assert gateway.count("initialize") == 0


# --- version breadth --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", HANDSHAKE_VERSIONS)
async def test_legacy_negotiates_every_supported_handshake_version(
    protocol_version: str,
) -> None:
    """Every handshake version SDK 2 still accepts works through ``McpClient``.

    ``2024-11-05`` and ``2025-03-26`` are covered nowhere else. All four
    negotiate identically here -- the server's counter-offer is honoured and
    stamped onto every later request -- so no version needs a carve-out.
    """
    server = PinnedVersionServer(protocol_version)
    gateway = RecordingGateway(server.build_app())
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="legacy") as client:
            result = await client.call_tool("add", {"a": 2, "b": 3})
            session_id = await client.get_session_id()

    assert result.structured_content == {"result": 5}
    assert server.initialize_count == 1
    assert session_id == "session-1"
    assert negotiated_version(gateway, "tools/call") == protocol_version
    assert gateway.for_rpc("tools/call")[0].session_id == "session-1"


# --- lifecycle --------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_is_cached_until_force_refresh() -> None:
    """Discovery is fetched once per client and re-queried only on demand."""
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        async with connected_client(url) as client:
            first = await client.list_tools()
            second = await client.list_tools()
            assert gateway.count("tools/list") == 1

            refreshed = await client.list_tools(force_refresh=True)

    assert first is second
    assert sorted(tool.name for tool in refreshed.tools) == ["add", "multiply"]
    assert gateway.count("tools/list") == 2


@pytest.mark.asyncio
async def test_dispose_then_reuse_reinitializes_the_client() -> None:
    """Disposal releases everything, and the next call rebuilds a working client."""
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        async with connected_client(url) as client:
            await client.call_tool("add", {"a": 1, "b": 1})
            await client.dispose()

            assert not client.is_client_initialized
            assert await client.get_session_id() is None

            result = await client.call_tool("add", {"a": 5, "b": 5})

            assert result.structured_content == {"result": 10}
            assert client.is_client_initialized

    assert gateway.count("initialize") == 2
    first_call, second_call = gateway.for_rpc("tools/call")
    assert first_call.session_id != second_call.session_id


@pytest.mark.asyncio
async def test_tool_built_by_the_factory_invokes_over_real_http() -> None:
    """Drive the whole seam: factory -> LangChain tool -> McpClient -> the wire.

    Every other tool test substitutes ``MagicMock(spec=McpClient)``, so nothing
    in pytest connected the factory to a real server. That gap let a silent
    serialization regression ship: ``_normalize_tool_result`` kept using a plain
    ``model_dump()`` after SDK 2.0 renamed the model attributes, rewriting
    ``mimeType`` to ``mime_type`` for every non-text block handed to the model.
    """
    from uipath_langchain.agent.tools.mcp import create_mcp_tools

    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        async with connected_client(url) as client:
            tools = await create_mcp_tools(make_resource_config(), client)
            add_tool = next(tool for tool in tools if tool.name == "add")
            result = await add_tool.ainvoke({"a": 2, "b": 3})

    blocks = result if isinstance(result, list) else [result]
    assert [block.get("text") for block in blocks] == ["5"]
    assert gateway.count("tools/call") == 1


@pytest.mark.asyncio
async def test_factory_tool_hands_non_text_blocks_over_in_wire_shape() -> None:
    """A non-text block reaches the model camelCased, through the factory path.

    Text blocks serialize identically under either dump mode, so only a
    non-text block can catch a snake_case regression. Driving it through
    ``build_mcp_tool`` covers the serializer the factory actually installs.
    """
    from mcp.types import CallToolResult, ImageContent

    from uipath_langchain.agent.tools.mcp.mcp_tool import _normalize_tool_result

    normalized = _normalize_tool_result(
        CallToolResult(
            content=[ImageContent(type="image", data="Zm9v", mimeType="image/png")]
        )
    )

    assert normalized == [{"type": "image", "data": "Zm9v", "mimeType": "image/png"}], (
        "non-text blocks must keep their wire spelling for the model"
    )


@pytest.mark.asyncio
async def test_dispose_clears_the_tool_cache() -> None:
    """A resumed run must not serve a tool list captured before disposal.

    ``dispose()`` clearing ``_tools_cache`` had no assertion anywhere, so a
    stale list surviving dispose/reuse would have passed the suite.
    """
    gateway = RecordingGateway(build_sdk_app())
    async with serve(gateway) as url:
        with patched_sdk(url):
            client = make_client()
            try:
                await client.list_tools()
                await client.list_tools()
                assert gateway.count("tools/list") == 1, "cache did not hold"
                await client.dispose()
                await client.list_tools()
                assert gateway.count("tools/list") == 2, "cache survived dispose"
            finally:
                await client.dispose()


@pytest.mark.asyncio
async def test_session_info_factory_receives_the_resolved_mcp_server() -> None:
    """The factory is handed the resolved server, not just a URL.

    ``SessionInfoDebugStateFactory`` downstream keys its debug-state path on the
    server's ``slug``, so losing that argument would break persistence there
    while every test here still passed.
    """
    seen: list[Any] = []

    class _RecordingFactory(SessionInfoFactory):
        def create_session(self, mcp_server: Any) -> SessionInfo:
            seen.append(mcp_server)
            return SessionInfo()

    async with serve(build_sdk_app()) as url:
        async with connected_client(
            url, session_info_factory=_RecordingFactory()
        ) as client:
            await client.list_tools()

    assert len(seen) == 1
    assert seen[0].slug
    assert seen[0].mcp_url


@pytest.mark.asyncio
async def test_recovery_reuses_the_same_http_client() -> None:
    """Replacing a lost session must not rebuild the authenticated HTTP client.

    Recovery reuses the client precisely so a reconnect costs no TLS handshake
    or token resolution. That was previously only a docstring claim.
    """
    gateway = RecordingGateway(build_sdk_app(), fault_on_tool_call=1)
    async with serve(gateway) as url:
        async with connected_client(url, protocol_mode="legacy") as client:
            await client.list_tools()
            first_http_client = client._http_client
            await client.call_tool("add", {"a": 2, "b": 3})

            assert gateway.count("initialize") == 2, "no session replacement happened"
            assert client._http_client is first_http_client


@pytest.mark.asyncio
async def test_server_is_resolved_by_name_and_execution_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy SDK lookup must pass the display name and execution folder.

    `uipath debug` applies resource bindings after the graph is built, which is
    why this lookup is deferred to the first call. Dropping either argument would
    resolve the wrong server -- or the right one in the wrong folder -- while
    every wire-level assertion here still passed. The folder comes from
    ``UIPATH_FOLDER_PATH``, which the runtime sets per job.
    """
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "/Shared/SomeExecutionFolder")

    async with serve(build_sdk_app()) as url:
        async with connected_client(url) as client:
            await client.list_tools()

    assert len(SDK_LOOKUPS) == 1
    assert SDK_LOOKUPS[0] == {
        "name": make_resource_config().name,
        "folder_path": "/Shared/SomeExecutionFolder",
    }
