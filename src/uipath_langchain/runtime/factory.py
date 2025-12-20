import asyncio
import os
from typing import Any, AsyncContextManager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_ancestor_spans,
    get_current_span,
)
from uipath.core.tracing import UiPathSpanUtils, UiPathTraceManager
from uipath.platform.resume_triggers import (
    UiPathResumeTriggerHandler,
)
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._tracing import _instrument_traceable_attributes
from uipath_langchain.runtime.config import LangGraphConfig
from uipath_langchain.runtime.errors import LangGraphErrorCode, LangGraphRuntimeError
from uipath_langchain.runtime.graph import LangGraphLoader
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime
from uipath_langchain.runtime.storage import SqliteResumableStorage


class UiPathLangGraphRuntimeFactory:
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

        self._graph_cache: dict[str, CompiledStateGraph[Any, Any, Any, Any]] = {}
        self._graph_loaders: dict[str, LangGraphLoader] = {}
        self._graph_lock = asyncio.Lock()

        self._setup_instrumentation(self.context.trace_manager)

    def _setup_instrumentation(self, trace_manager: UiPathTraceManager | None) -> None:
        """Setup tracing and instrumentation."""
        _instrument_traceable_attributes()
        LangChainInstrumentor().instrument()
        UiPathSpanUtils.register_current_span_provider(get_current_span)
        UiPathSpanUtils.register_current_span_ancestors_provider(get_ancestor_spans)

    def _get_connection_string(self) -> str:
        """Get the database connection string."""
        if self.context.runtime_dir and self.context.state_file:
            path = os.path.join(self.context.runtime_dir, self.context.state_file)
            if not self.context.resume and self.context.job_id is None:
                # If not resuming and no job id, delete the previous state file
                if os.path.exists(path):
                    os.remove(path)
            os.makedirs(self.context.runtime_dir, exist_ok=True)
            return path

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
        return self._config

    async def _load_graph(
        self, entrypoint: str
    ) -> StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any]:
        """
        Load a graph for the given entrypoint.

        Args:
            entrypoint: Name of the graph to load

        Returns:
            The loaded StateGraph or CompiledStateGraph

        Raises:
            LangGraphRuntimeError: If graph cannot be loaded
        """
        config = self._load_config()
        if not config.exists:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        if entrypoint not in config.graphs:
            available = ", ".join(config.entrypoints)
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_NOT_FOUND,
                "Graph not found",
                f"Graph '{entrypoint}' not found. Available: {available}",
                UiPathErrorCategory.DEPLOYMENT,
            )

        path = config.graphs[entrypoint]
        graph_loader = LangGraphLoader.from_path_string(entrypoint, path)

        self._graph_loaders[entrypoint] = graph_loader

        try:
            return await graph_loader.load()

        except ImportError as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_IMPORT_ERROR,
                "Graph import failed",
                f"Failed to import graph '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except TypeError as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_TYPE_ERROR,
                "Invalid graph type",
                f"Graph '{entrypoint}' is not a valid StateGraph or CompiledStateGraph: {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except ValueError as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_VALUE_ERROR,
                "Invalid graph value",
                f"Invalid value in graph '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_LOAD_ERROR,
                "Failed to load graph",
                f"Unexpected error loading graph '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    async def _compile_graph(
        self,
        graph: StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any],
        memory: AsyncSqliteSaver,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """
        Compile a graph with the given memory/checkpointer.

        Args:
            graph: The graph to compile (StateGraph or already compiled)
            memory: Checkpointer to use for compiled graph

        Returns:
            The compiled StateGraph
        """
        builder = graph.builder if isinstance(graph, CompiledStateGraph) else graph

        return builder.compile(checkpointer=memory)

    async def _resolve_and_compile_graph(
        self, entrypoint: str, memory: AsyncSqliteSaver
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """
        Resolve a graph from configuration and compile it.
        Results are cached for reuse across multiple runtime instances.

        Args:
            entrypoint: Name of the graph to resolve
            memory: Checkpointer to use for compiled graph

        Returns:
            The compiled StateGraph ready for execution

        Raises:
            LangGraphRuntimeError: If resolution or compilation fails
        """
        async with self._graph_lock:
            if entrypoint in self._graph_cache:
                return self._graph_cache[entrypoint]

            loaded_graph = await self._load_graph(entrypoint)

            compiled_graph = await self._compile_graph(loaded_graph, memory)

            self._graph_cache[entrypoint] = compiled_graph

            return compiled_graph

    def discover_entrypoints(self) -> list[str]:
        """
        Discover all graph entrypoints.

        Returns:
            List of graph names that can be used as entrypoints
        """
        config = self._load_config()
        if not config.exists:
            return []
        return config.entrypoints

    async def discover_runtimes(self) -> list[UiPathRuntimeProtocol]:
        """
        Discover runtime instances for all entrypoints.

        Returns:
            List of LangGraphRuntime instances, one per entrypoint
        """
        entrypoints = self.discover_entrypoints()
        memory = await self._get_memory()

        runtimes: list[UiPathRuntimeProtocol] = []
        for entrypoint in entrypoints:
            compiled_graph = await self._resolve_and_compile_graph(entrypoint, memory)

            runtime = await self._create_runtime_instance(
                compiled_graph=compiled_graph,
                runtime_id=entrypoint,
                entrypoint=entrypoint,
            )
            runtimes.append(runtime)

        return runtimes

    async def _create_runtime_instance(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """
        Create a runtime instance from a compiled graph.

        Args:
            compiled_graph: The compiled graph
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Graph entrypoint name

        Returns:
            Configured runtime instance
        """
        base_runtime = UiPathLangGraphRuntime(
            graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

        memory = await self._get_memory()
        storage = SqliteResumableStorage(memory)
        trigger_manager = UiPathResumeTriggerHandler()

        return UiPathResumableRuntime(
            delegate=base_runtime,
            storage=storage,
            trigger_manager=trigger_manager,
            runtime_id=runtime_id,
        )

    async def new_runtime(
        self, entrypoint: str, runtime_id: str
    ) -> UiPathRuntimeProtocol:
        """
        Create a new LangGraph runtime instance.

        Args:
            entrypoint: Graph name from langgraph.json
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Configured runtime instance with compiled graph
        """
        # Get shared memory instance
        memory = await self._get_memory()

        compiled_graph = await self._resolve_and_compile_graph(entrypoint, memory)

        return await self._create_runtime_instance(
            compiled_graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        for loader in self._graph_loaders.values():
            await loader.cleanup()

        self._graph_loaders.clear()
        self._graph_cache.clear()

        if self._memory_cm is not None:
            await self._memory_cm.__aexit__(None, None, None)
            self._memory_cm = None
            self._memory = None
