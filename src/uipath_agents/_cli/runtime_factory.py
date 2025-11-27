import asyncio
import os
from typing import AsyncContextManager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_ancestor_spans,
    get_current_span,
)
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from uipath.core.tracing import UiPathSpanUtils
from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeProtocol
from uipath_langchain._cli._runtime._factory import LangGraphRuntimeFactory
from uipath_langchain._cli._utils._graph import LangGraphConfig
from uipath_langchain._tracing import _instrument_traceable_attributes

from .runtime import create_agent_langgraph_runtime


def agents_telemetry_config():
    os.environ.setdefault("OTEL_SERVICE_NAME", "uipath-agents")

    _instrument_traceable_attributes()
    AsyncioInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
    SQLite3Instrumentor().instrument()
    LangChainInstrumentor().instrument()


class AgentRuntimeFactory(LangGraphRuntimeFactory):
    """Factory for creating Agent runtimes from agent.json configuration."""

    def __init__(
        self,
        context: UiPathRuntimeContext,
    ):
        """
        Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
        self.context = context
        self._config: LangGraphConfig | None = None
        self._memory: AsyncSqliteSaver | None = None
        self._memory_cm: AsyncContextManager[AsyncSqliteSaver] | None = None
        self._memory_lock = asyncio.Lock()
        agents_telemetry_config()
        UiPathSpanUtils.register_current_span_provider(get_current_span)
        UiPathSpanUtils.register_current_span_ancestors_provider(get_ancestor_spans)

    def discover_entrypoints(self) -> list[str]:
        """
        Discover all graph entrypoints from langgraph.json.

        Returns:
            List of graph names that can be used as entrypoints
        """

        return ["agent.json"]

    async def discover_runtimes(self) -> list[UiPathRuntimeProtocol]:
        """
        Discover runtime instances for all entrypoints.

        Returns:
            List of LangGraphScriptRuntime instances, one per entrypoint
        """
        entrypoints = self.discover_entrypoints()
        memory = await self._get_memory()

        runtimes: list[UiPathRuntimeProtocol] = []
        for entrypoint in entrypoints:
            runtime = await create_agent_langgraph_runtime(
                entrypoint, self.context, memory
            )
            runtimes.append(runtime)

        return runtimes

    async def new_runtime(
        self, entrypoint: str, runtime_id: str
    ) -> UiPathRuntimeProtocol:
        """
        Create a new LangGraph runtime instance.

        Args:
            entrypoint: Graph name from langgraph.json
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Configured LangGraphScriptRuntime instance
        """
        # Get shared memory instance
        memory = await self._get_memory()

        return create_agent_langgraph_runtime(runtime_id, self.context, memory)

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        if self._memory_cm is not None:
            await self._memory_cm.__aexit__(None, None, None)
            self._memory_cm = None
            self._memory = None
