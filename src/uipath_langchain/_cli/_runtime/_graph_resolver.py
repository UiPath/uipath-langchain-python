import asyncio
from typing import Any, Awaitable, Callable, Optional

from langgraph.graph.state import CompiledStateGraph, StateGraph
from uipath._cli._runtime._contracts import UiPathErrorCategory, UiPathErrorCode

from .._utils._graph import GraphConfig, LangGraphConfig
from ._exception import LangGraphErrorCode, LangGraphRuntimeError


class LangGraphJsonResolver:
    def __init__(self, entrypoint: Optional[str] = None) -> None:
        self.entrypoint = entrypoint
        self.graph_config: Optional[GraphConfig] = None
        self._lock = asyncio.Lock()
        self._graph_cache: Optional[StateGraph[Any, Any, Any]] = None
        self._resolving: bool = False

    async def __call__(self) -> StateGraph[Any, Any, Any]:
        # Fast path: if already resolved, return immediately without locking
        if self._graph_cache is not None:
            return self._graph_cache

        # Slow path: acquire lock and resolve
        async with self._lock:
            # Double-check after acquiring lock (another coroutine may have resolved it)
            if self._graph_cache is not None:
                return self._graph_cache

            self._graph_cache = await self._resolve(self.entrypoint)
            return self._graph_cache

    async def _resolve(self, entrypoint: Optional[str]) -> StateGraph[Any, Any, Any]:
        config = LangGraphConfig()
        if not config.exists:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.CONFIG_MISSING,
                "Invalid configuration",
                "Failed to load configuration",
                UiPathErrorCategory.DEPLOYMENT,
            )

        try:
            config.load_config()
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.CONFIG_INVALID,
                "Invalid configuration",
                f"Failed to load configuration: {str(e)}",
                UiPathErrorCategory.DEPLOYMENT,
            ) from e

        # Determine entrypoint if not provided
        graphs = config.graphs
        if not entrypoint and len(graphs) == 1:
            entrypoint = graphs[0].name
        elif not entrypoint:
            graph_names = ", ".join(g.name for g in graphs)
            raise LangGraphRuntimeError(
                UiPathErrorCode.ENTRYPOINT_MISSING,
                "Entrypoint required",
                f"Multiple graphs available. Please specify one of: {graph_names}.",
                UiPathErrorCategory.DEPLOYMENT,
            )

        # Get the specified graph
        self.graph_config = config.get_graph(entrypoint)
        if not self.graph_config:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_NOT_FOUND,
                "Graph not found",
                f"Graph '{entrypoint}' not found.",
                UiPathErrorCategory.DEPLOYMENT,
            )
        try:
            loaded_graph = await self.graph_config.load_graph()
            return (
                loaded_graph.builder
                if isinstance(loaded_graph, CompiledStateGraph)
                else loaded_graph
            )
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

    async def cleanup(self):
        """Clean up resources"""
        async with self._lock:
            if self.graph_config:
                await self.graph_config.cleanup()
                self.graph_config = None
            self._graph_cache = None


AsyncResolver = Callable[[], Awaitable[StateGraph[Any, Any, Any]]]


class LangGraphJsonResolverContext:
    """
    Async context manager wrapping LangGraphJsonResolver.
    Returns a callable that can be passed directly as AsyncResolver to LangGraphRuntime.
    Thread-safe and reuses the same resolved graph across concurrent executions.
    """

    def __init__(self, entrypoint: Optional[str] = None) -> None:
        self._resolver = LangGraphJsonResolver(entrypoint)

    async def __aenter__(self) -> AsyncResolver:
        # Return a callable that safely reuses the cached graph
        async def resolver_callable() -> StateGraph[Any, Any, Any]:
            return await self._resolver()

        return resolver_callable

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._resolver.cleanup()
