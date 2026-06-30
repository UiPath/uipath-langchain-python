"""A2A (Agent-to-Agent) tools."""

import os

# The a2a-sdk auto-instruments its JSON-RPC transport with OpenTelemetry spans
# (``a2a.client.transports.jsonrpc.JsonRpcTransport.*``). Those internal spans
# surface as noisy, unparented nodes in the Execution Trace and are not a
# meaningful representation of the call. Disable that instrumentation before the
# SDK is imported (it reads this variable at import time), so the single A2A
# span emitted by ``a2a_tool`` is the only node for the call. ``setdefault``
# preserves an explicit opt-in when the variable is already set.
os.environ.setdefault("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", "false")

from .a2a_tool import (  # noqa: E402
    A2aClient,
    create_a2a_tools_and_clients,
    open_a2a_tools,
)

__all__ = [
    "A2aClient",
    "create_a2a_tools_and_clients",
    "open_a2a_tools",
]
