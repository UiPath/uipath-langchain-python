import json
import os

print("Checking MCP-over-HTTP protocol version matrix...")

# Operands from input.json; the remote add tool must return their sum.
EXPECTED_SUM = "5"

# The low-level ClientSession only speaks the legacy handshake versions, so a
# modern-only server is expected to fail with METHOD_NOT_FOUND. This leg should
# flip to a success once modern discovery is supported.
METHOD_NOT_FOUND = -32601

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

by_version = {r["protocol_version"]: r for r in results}
for version in ("2025-11-25", "2025-06-18", "2026-07-28"):
    assert version in by_version, f"No result recorded for protocol version {version}"


def check_supported(version: str, expected_tool: str) -> None:
    """A negotiable version must connect, issue a session, and call a tool."""
    leg = by_version[version]
    assert leg["supported"], (
        f"{version} should be supported but failed: {leg.get('error_message')}"
    )
    negotiated = leg["negotiated_version"]
    assert negotiated == version, (
        f"{version} leg negotiated {negotiated} instead of {version}"
    )
    # Every legacy handshake version still uses mcp-session-id, and UiPath's
    # SessionInfo adapter must have captured it over real HTTP.
    assert leg["session_id_issued"], f"{version} did not persist a session id"
    assert expected_tool in leg["tools"], (
        f"{version} did not discover the '{expected_tool}' tool: {leg['tools']}"
    )
    assert leg["tool_result"] == EXPECTED_SUM, (
        f"{version} tool returned {leg['tool_result']!r}, expected {EXPECTED_SUM!r}"
    )
    print(f"{version}: negotiated, session persisted, tool call returned {EXPECTED_SUM}")


check_supported("2025-11-25", "multiply")
check_supported("2025-06-18", "add")

# 2026-07-28 is modern-discovery only and unreachable from ClientSession today.
modern = by_version["2026-07-28"]
assert not modern["supported"], (
    "2026-07-28 unexpectedly succeeded. If modern discovery is now supported, "
    "this assertion should be inverted to expect a successful negotiation."
)
assert modern["error_code"] == METHOD_NOT_FOUND, (
    f"2026-07-28 failed with code {modern['error_code']}, "
    f"expected METHOD_NOT_FOUND ({METHOD_NOT_FOUND})"
)
assert not modern["session_id_issued"], (
    "2026-07-28 must not establish a session; the handshake never completes"
)
print(
    "2026-07-28: correctly unsupported "
    f"(code {modern['error_code']}, no session established)"
)

assert output_content["supported_versions"] == ["2025-11-25", "2025-06-18"], (
    f"Unexpected supported versions: {output_content['supported_versions']}"
)
assert output_content["unsupported_versions"] == ["2026-07-28"], (
    f"Unexpected unsupported versions: {output_content['unsupported_versions']}"
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

# SessionInfo.protocol_version is new in MCP 2 and the transport reads it
# directly, so a subclass calling super().__init__() must end up with it set.
assert agents_api["session_info_super_init"], (
    "SessionInfo subclass did not inherit protocol_version from super().__init__()"
)


def check_leg(name: str, expected_tools: list[str]) -> dict:
    """A downstream-shaped call must build tools, run one, and dispose cleanly."""
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
    print(f"{name}: tools {leg['tools']}, tool call returned {EXPECTED_SUM}, disposed")
    return leg


# Production shape: no session_info_factory, terminate_on_close=True, dynamic
# discovery -- which reads the SDK's snake_case Tool.input_schema/output_schema.
check_leg("production", ["add", "multiply"])

# Playground shape: SessionInfoFactory subclass, terminate_on_close=False,
# cached discovery with refresh_schema_before_call left at its default.
playground = check_leg("playground", ["add"])
resumed = check_leg("playground_resumed", ["add"])

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

print("uipath-agents-python API compatibility validation passed")
