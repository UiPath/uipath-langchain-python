import os

from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeProtocol
from uipath_langchain._cli._runtime._factory import LangGraphRuntimeFactory

from ..constants import AGENT_ENTRYPOINT
from .runtime import create_agent_langgraph_runtime

_telemetry_initialized = False


def _configure_agents_telemetry() -> None:
    """Configure telemetry for agents. Idempotent - only runs once."""
    global _telemetry_initialized
    if _telemetry_initialized:
        return

    os.environ.setdefault("OTEL_SERVICE_NAME", "uipath-agents")
    AsyncioInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
    SQLite3Instrumentor().instrument()
    _telemetry_initialized = True


class AgentRuntimeFactory(LangGraphRuntimeFactory):
    """Factory for creating Agent runtimes from agent.json configuration."""

    def __init__(self, context: UiPathRuntimeContext) -> None:
        """Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
        super().__init__(context)
        _configure_agents_telemetry()

    def discover_entrypoints(self) -> list[str]:
        """Discover the Agent entrypoint agent.json"""
        return [AGENT_ENTRYPOINT]

    async def discover_runtimes(self) -> list[UiPathRuntimeProtocol]:
        """Discover runtime instances for all entrypoints.

        Returns:
            List of AgentLangGraphRuntime instances
        """
        entrypoints = self.discover_entrypoints()
        memory = await self._get_memory()

        runtimes: list[UiPathRuntimeProtocol] = []
        for entrypoint in entrypoints:
            runtime = create_agent_langgraph_runtime(
                entrypoint, entrypoint, self.context, memory
            )
            runtimes.append(runtime)

        return runtimes

    async def new_runtime(
        self, entrypoint: str, runtime_id: str
    ) -> UiPathRuntimeProtocol:
        """Create a new Agent runtime instance.

        Args:
            entrypoint: Agent entrypoint (agent.json)
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Configured AgentLangGraphRuntime instance
        """
        memory = await self._get_memory()
        return create_agent_langgraph_runtime(
            runtime_id, entrypoint, self.context, memory
        )
