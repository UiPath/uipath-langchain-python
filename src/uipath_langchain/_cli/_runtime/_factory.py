import asyncio
import os
from typing import AsyncContextManager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_ancestor_spans,
    get_current_span,
)
from uipath.core.tracing import UiPathSpanUtils
from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeProtocol

from uipath_langchain._tracing import _instrument_traceable_attributes

from .._utils._graph import LangGraphConfig
from ._runtime import LangGraphScriptRuntime


class LangGraphRuntimeFactory:
    """Factory for creating LangGraph runtimes from langgraph.json configuration."""

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
        _instrument_traceable_attributes()
        LangChainInstrumentor().instrument()
        UiPathSpanUtils.register_current_span_provider(get_current_span)
        UiPathSpanUtils.register_current_span_ancestors_provider(get_ancestor_spans)

    def _get_connection_string(self) -> str:
        """Get the database connection string with same logic as get_memory."""
        if self.context.runtime_dir and self.context.state_file:
            path = os.path.join(self.context.runtime_dir, self.context.state_file)
            if not self.context.resume and self.context.job_id is None:
                # If not resuming and no job id, delete the previous state file
                if os.path.exists(path):
                    os.remove(path)
            os.makedirs(self.context.runtime_dir, exist_ok=True)
            return path

        # Default path
        default_path = os.path.join("__uipath", "state.db")
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        return default_path

    async def _get_memory(self) -> AsyncSqliteSaver:
        """Get or create the shared memory instance."""
        async with self._memory_lock:
            if self._memory is None:
                connection_string = self._get_connection_string()
                self._memory_cm = AsyncSqliteSaver.from_conn_string(connection_string)
                self._memory = await self._memory_cm.__aenter__()
                await self._memory.setup()
        return self._memory

    def _load_config(self) -> LangGraphConfig:
        """Load langgraph.json configuration."""
        if self._config is None:
            self._config = LangGraphConfig()
            if self._config.exists:
                self._config.load_config()
        return self._config

    def discover_entrypoints(self) -> list[str]:
        """
        Discover all graph entrypoints from langgraph.json.

        Returns:
            List of graph names that can be used as entrypoints
        """
        config = self._load_config()
        if not config.exists:
            return []

        return [graph.name for graph in config.graphs]

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
            runtime = LangGraphScriptRuntime(
                runtime_id=entrypoint,
                memory=memory,
                entrypoint=entrypoint,
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

        return LangGraphScriptRuntime(runtime_id, memory=memory, entrypoint=entrypoint)

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        if self._memory_cm is not None:
            await self._memory_cm.__aexit__(None, None, None)
            self._memory_cm = None
            self._memory = None
