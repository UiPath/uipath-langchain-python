import json
import os

print("Checking MCP-over-HTTP protocol version matrix...")

# Operands from input.json; the remote add tool must return their sum.
EXPECTED_SUM = "5"
MODERN_VERSION = "2026-07-28"

# Check NuGet package
uipath_dir = ".uipath"
assert os.path.exists(uipath_dir), "NuGet package directory (.uipath) not found"

nupkg_files = [f for f in os.listdir(uipath_dir) if f.endswith(".nupkg")]
assert nupkg_files, "NuGet package file (.nupkg) not found in .uipath directory"

print(f"NuGet package found: {nupkg_files[0]}")

# Check agent output file
output_file = "__uipath/output.json"
assert os.path.isfile(output_file), "Agent output file not found"

print("Agent output file found")

with open(output_file, "r", encoding="utf-8") as f:
    output_data = json.load(f)

status = output_data.get("status")
assert status == "successful", f"Agent execution failed with status: {status}"

print("Agent execution status: successful")

assert "output" in output_data, "Missing 'output' field in agent response"
output_content = output_data["output"]

for field in ("results", "supported_versions", "unsupported_versions"):
    assert field in output_content, f"Missing '{field}' field in output"

results = output_content["results"]
assert isinstance(results, list) and results, "Results field is empty or not a list"

by_label = {r["label"]: r for r in results}
EXPECTED_LEGS = (
    "legacy-sdk-server",
    "legacy-pinned-2025-06-18",
    "modern-sdk-server",
    "modern-only-endpoint",
    "auto-sdk-server",
    "auto-pinned-2025-06-18",
)
for label in EXPECTED_LEGS:
    assert label in by_label, f"No result recorded for leg {label}"


def check_leg_negotiated(
    label: str,
    version: str,
    era: str,
    expected_tool: str,
    *,
    server_session: bool,
) -> None:
    """Every leg must negotiate its era, discover tools, and run one."""
    leg = by_label[label]
    assert leg["supported"], (
        f"{label} should be supported but failed: {leg.get('error_message')}"
    )
    assert leg["era"] == era, f"{label} resolved era {leg['era']}, expected {era}"
    assert leg["negotiated_version"] == version, (
        f"{label} negotiated {leg['negotiated_version']} instead of {version}"
    )
    # A legacy session ID comes from the server; a modern one is minted by the
    # client for routing, so the wire must show no server-assigned session.
    assert leg["server_session_id_seen"] is server_session, (
        f"{label} server_session_id_seen={leg['server_session_id_seen']}, "
        f"expected {server_session}"
    )
    assert expected_tool in leg["tools"], (
        f"{label} did not discover the '{expected_tool}' tool: {leg['tools']}"
    )
    assert leg["tool_result"] == EXPECTED_SUM, (
        f"{label} tool returned {leg['tool_result']!r}, expected {EXPECTED_SUM!r}"
    )
    print(f"{label}: {era} {version}, tool call returned {EXPECTED_SUM}")


# Legacy era: the server mints and returns a session ID.
check_leg_negotiated(
    "legacy-sdk-server", "2025-11-25", "legacy", "multiply", server_session=True
)
check_leg_negotiated(
    "legacy-pinned-2025-06-18", "2025-06-18", "legacy", "add", server_session=True
)

# Modern era: session IDs are gone from the protocol entirely. The modern-only
# endpoint refuses the handshake, so this cannot be passing via legacy fallback.
check_leg_negotiated(
    "modern-sdk-server", MODERN_VERSION, "modern", "multiply", server_session=False
)
check_leg_negotiated(
    "modern-only-endpoint", MODERN_VERSION, "modern", "add", server_session=False
)

# auto resolves a different era per server, preferring modern and falling back.
check_leg_negotiated(
    "auto-sdk-server", MODERN_VERSION, "modern", "multiply", server_session=False
)
check_leg_negotiated(
    "auto-pinned-2025-06-18", "2025-06-18", "legacy", "add", server_session=True
)

assert output_content["unsupported_versions"] == [], (
    f"Every leg should negotiate now: {output_content['unsupported_versions']}"
)

# --- gateway affinity -------------------------------------------------------
# 2026-07-28 removes mcp-session-id, so the UiPath affinity ID is what lets a
# gateway keep routing to one warm serverless instance.
print("Checking modern-era instance affinity...")

affinity = output_content.get("affinity")
assert affinity, "Missing 'affinity' field in output"
assert affinity["tool_results"] == [EXPECTED_SUM, EXPECTED_SUM], (
    f"Affinity leg tool results were {affinity['tool_results']}"
)
ids = affinity["affinity_ids"]
assert len(ids) == 2 and ids[0] and ids[0] == ids[1], (
    f"The affinity ID must persist across clients, got {ids}"
)
assert affinity["instances"] and len(affinity["instances"]) == 1, (
    f"Requests spread across instances {affinity['instances']}; affinity failed"
)
# The client mints the ID before negotiating, so even discovery is routable --
# unlike a server-assigned session, which cannot reach the first request.
assert affinity["first_request_pinned"], (
    "The first request was not pinned; the affinity ID reached the gateway late"
)
assert affinity["requests"] > 0 and affinity["unpinned_requests"] == 0, (
    f"{affinity['unpinned_requests']} of {affinity['requests']} requests reached "
    "the gateway with no affinity header, so it had to route them blind"
)
print(
    f"Affinity ID {ids[0]} pinned {affinity['instances'][0]} for all "
    f"{affinity['requests']} requests across both clients"
)

print("Protocol version matrix validation passed")

# --- uipath-agents-python compatibility -------------------------------------
# Everything below pins the MCP API surface consumed by uipath-agents-python.
# A failure here means that repository breaks on its next dependency bump.

print("Checking uipath-agents-python API compatibility...")

assert "agents_api" in output_content, "Missing 'agents_api' field in output"
agents_api = output_content["agents_api"]
assert agents_api, "Downstream-compatibility leg produced no result"

assert agents_api["imports"] == [
    "McpClient",
    "SessionInfo",
    "SessionInfoFactory",
    "create_mcp_tools_and_clients",
], f"Unexpected downstream import list: {agents_api['imports']}"

# SessionInfo.protocol_version is retained for compatibility even though nothing
# reads it now, so a subclass calling super().__init__() must still inherit it.
assert agents_api["session_info_super_init"], (
    "SessionInfo subclass did not inherit protocol_version from super().__init__()"
)


def check_leg(
    name: str,
    expected_tools: list[str],
    *,
    expected_version: str | None = None,
    server_session: bool = True,
) -> dict:
    """A downstream-shaped call must build tools, run one, and dispose cleanly.

    Args:
        name: Field on the agents_api result holding this leg.
        expected_tools: Tool names the leg must have built.
        expected_version: Protocol version the live session must have settled
            on, or None to skip the check.
        server_session: Whether the server is allowed to assign an
            ``mcp-session-id``. False for modern legs, which have no session.
    """
    leg = agents_api[name]
    assert leg["error_type"] is None, (
        f"{name} leg failed: {leg['error_type']}: {leg['error_message']}"
    )
    assert leg["tools"] == expected_tools, (
        f"{name} leg built tools {leg['tools']}, expected {expected_tools}"
    )
    assert leg["tool_result"] == EXPECTED_SUM, (
        f"{name} leg tool returned {leg['tool_result']!r}, expected {EXPECTED_SUM!r}"
    )
    assert leg["session_id"], f"{name} leg did not establish a session id"
    assert leg["disposed"], f"{name} leg did not dispose its McpClient"
    if expected_version is not None:
        assert leg["negotiated_version"] == expected_version, (
            f"{name} leg negotiated {leg['negotiated_version']!r}, expected "
            f"{expected_version!r}"
        )
    assert leg["server_session_issued"] is server_session, (
        f"{name} leg server_session_issued={leg['server_session_issued']}, "
        f"expected {server_session}"
    )
    print(f"{name}: tools {leg['tools']}, tool call returned {EXPECTED_SUM}, disposed")
    return leg


# Production shape: no session_info_factory, terminate_on_close=True, dynamic
# discovery -- which reads the SDK's snake_case Tool.input_schema/output_schema.
check_leg("production", ["add", "multiply"], expected_version="2025-11-25")

# Playground shape: SessionInfoFactory subclass, terminate_on_close=False,
# cached discovery with refresh_schema_before_call left at its default.
playground = check_leg("playground", ["add"], expected_version="2025-11-25")
resumed = check_leg("playground_resumed", ["add"], expected_version="2025-11-25")

# The resumed session must speak the version it was originally negotiated at.
# The session ID surviving is only half the contract: probing candidate versions
# instead settles on the oldest handshake version and silently downgrades every
# later request, which no session-ID assertion can catch.
assert resumed["negotiated_version"] == playground["negotiated_version"], (
    f"Resumed session negotiated {resumed['negotiated_version']!r} but the "
    f"session it resumed was negotiated at {playground['negotiated_version']!r}; "
    "the resumed connection was silently downgraded"
)

assert agents_api["debug_state_writes"] >= 1, (
    "The SessionInfo subclass never persisted a session id"
)
persisted = agents_api["persisted_session_id"]
assert persisted == playground["session_id"], (
    f"Persisted session id {persisted!r} does not match the playground leg's "
    f"{playground['session_id']!r}"
)
# The second client must adopt the persisted session rather than start a new
# one; this is what playground mode relies on across runs.
assert agents_api["session_resumed"], (
    f"Second client did not resume the persisted session: "
    f"{resumed['session_id']!r} != {persisted!r}"
)
print(f"Persisted session {persisted} resumed by a second McpClient")

# --- McpClient on the modern era --------------------------------------------
# 2026-07-28 has no session identity at all, so no response on these legs may
# assign one; the ID in play is the client-minted affinity ID.
print("Checking McpClient on the 2026-07-28 era...")

check_leg(
    "modern",
    ["add", "multiply"],
    expected_version=MODERN_VERSION,
    server_session=False,
)
affinity_first = check_leg(
    "modern_affinity",
    ["add"],
    expected_version=MODERN_VERSION,
    server_session=False,
)
affinity_resumed = check_leg(
    "modern_affinity_resumed",
    ["add"],
    expected_version=MODERN_VERSION,
    server_session=False,
)

# Two clients sharing one SessionInfo must keep the same affinity ID, so a
# gateway routes both runs to the same warm instance.
assert agents_api["modern_affinity_id_reused"], (
    f"The second modern client sent affinity ID "
    f"{affinity_resumed['session_id']!r} instead of reusing "
    f"{affinity_first['session_id']!r}"
)
assert affinity_resumed["negotiated_version"] == affinity_first["negotiated_version"], (
    f"Resumed modern client negotiated "
    f"{affinity_resumed['negotiated_version']!r} instead of "
    f"{affinity_first['negotiated_version']!r}"
)
print(
    f"Affinity ID {affinity_first['session_id']} reused by a second modern "
    f"McpClient at {MODERN_VERSION}"
)

print("uipath-agents-python API compatibility validation passed")
