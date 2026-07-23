import asyncio
import hashlib
import os
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, AsyncContextManager, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_ancestor_spans,
    get_current_span,
)
from uipath.core.adapters import EvaluatorProtocol
from uipath.core.feature_flags import FeatureFlags
from uipath.core.tracing import UiPathSpanUtils, UiPathTraceManager
from uipath.platform import UiPath
from uipath.platform.resume_triggers import (
    UiPathResumeTriggerHandler,
)
from uipath.runtime import (
    HydrationPolicy,
    HydrationRuntime,
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
    UiPathRuntimeStorageProtocol,
    Workspace,
    WorkspaceHydrator,
    WorkspaceRegistryStore,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._tracing import _instrument_traceable_attributes
from uipath_langchain.deepagents.metadata import requires_managed_workspace
from uipath_langchain.governance import GovernanceCallbackHandler
from uipath_langchain.runtime.config import LangGraphConfig
from uipath_langchain.runtime.errors import LangGraphErrorCode, LangGraphRuntimeError
from uipath_langchain.runtime.graph import LangGraphLoader
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime
from uipath_langchain.runtime.storage import SqliteResumableStorage

_AGENT_TYPE_CODED = "uipath_coded"
_AGENT_FRAMEWORK = "langchain"
_MANAGED_WORKSPACE_HYDRATION_FEATURE_FLAG = "DeepAgentsWorkspaceHydration"


class _AsyncClosable(Protocol):
    """Public cleanup contract implemented by hydration platform services."""

    async def aclose(self) -> None: ...


class _UiPathHydrationRuntime(HydrationRuntime):
    """Hydration runtime that owns the platform services it lazily creates."""

    def __init__(
        self,
        *,
        dispose_platform_services: Callable[[], Awaitable[None]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._dispose_platform_services = dispose_platform_services

    async def dispose(self) -> None:
        """Dispose the delegate, workspace, and owned platform services."""
        try:
            await super().dispose()
        finally:
            await self._dispose_platform_services()


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
        if self.context.state_file_path is not None:
            return self.context.state_file_path

        if self.context.runtime_dir and self.context.state_file:
            path = os.path.join(self.context.runtime_dir, self.context.state_file)
            if (
                not self.context.resume
                and self.context.job_id is None
                and not self.context.keep_state_file
            ):
                # If not resuming and no job id, delete the previous state file
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass  # File may be held by another process
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
        self, entrypoint: str, **kwargs
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
        compiled = builder.compile(checkpointer=memory)
        bound_config = getattr(graph, "config", None)
        return compiled.with_config(bound_config) if bound_config else compiled

    async def _resolve_and_compile_graph(
        self, entrypoint: str, memory: AsyncSqliteSaver, **kwargs
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

            loaded_graph = await self._load_graph(entrypoint, **kwargs)
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

    async def get_settings(self) -> UiPathRuntimeFactorySettings | None:
        """Get the factory settings.

        Advertises this factory's ``agent_type`` and ``agent_framework``
        wire labels so hosts (governance audit, App Insights telemetry)
        can stamp them onto events without any host-side classification.
        """
        return UiPathRuntimeFactorySettings(
            agent_type=_AGENT_TYPE_CODED,
            agent_framework=_AGENT_FRAMEWORK,
        )

    async def get_storage(self) -> UiPathRuntimeStorageProtocol | None:
        """
        Get the runtime storage protocol instance.

        Returns:
            The storage protocol instance
        """
        memory = await self._get_memory()
        return SqliteResumableStorage(memory)

    async def _create_runtime_instance(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
        runtime_id: str,
        entrypoint: str,
        **kwargs,
    ) -> UiPathRuntimeProtocol:
        """
        Create a runtime instance from a compiled graph.

        Args:
            compiled_graph: The compiled graph
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Graph entrypoint name
            **kwargs: Forwarded factory kwargs. Recognized:
                ``evaluator`` (``EvaluatorProtocol``) — when present, the
                factory builds a :class:`GovernanceCallbackHandler` and
                hands it to the runtime via its ``callbacks`` arg.

        Returns:
            Configured runtime instance
        """
        memory = await self._get_memory()
        storage = SqliteResumableStorage(memory)
        trigger_manager = UiPathResumeTriggerHandler()

        evaluator: EvaluatorProtocol | None = kwargs.get("evaluator")
        callbacks: list[BaseCallbackHandler] | None = (
            [
                GovernanceCallbackHandler(
                    evaluator=evaluator,
                    agent_name=entrypoint,
                    session_id=runtime_id,
                )
            ]
            if evaluator is not None
            else None
        )

        workspace = (
            await self._create_managed_workspace(runtime_id)
            if self._uses_managed_workspace(compiled_graph)
            else None
        )

        base_runtime = UiPathLangGraphRuntime(
            graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            callbacks=callbacks,
            storage=storage,
            workspace_path=workspace.path if workspace is not None else None,
        )

        resumable_runtime: UiPathRuntimeProtocol = UiPathResumableRuntime(
            delegate=base_runtime,
            storage=storage,
            trigger_manager=trigger_manager,
            runtime_id=runtime_id,
        )

        if workspace is None:
            return resumable_runtime

        platform_services: tuple[_AsyncClosable, ...] | None = None

        def create_hydrator() -> WorkspaceHydrator:
            nonlocal platform_services
            sdk = UiPath()
            attachments = sdk.attachments
            jobs = sdk.jobs
            platform_services = (attachments, jobs)
            return WorkspaceHydrator(
                workspace_path=workspace.path,
                attachments=attachments,
                jobs=jobs,
                current_job_key=self.context.job_id,
                folder_key=self.context.folder_key,
            )

        async def dispose_platform_services() -> None:
            if platform_services is not None:
                await asyncio.gather(
                    *(service.aclose() for service in platform_services)
                )

        return _UiPathHydrationRuntime(
            delegate=resumable_runtime,
            workspace=workspace,
            hydrator_factory=create_hydrator,
            registry_store=WorkspaceRegistryStore(
                storage,
                runtime_id,
            ),
            policy=self._select_deep_agent_hydration_policy(),
            dispose_platform_services=dispose_platform_services,
        )

    @staticmethod
    def _uses_managed_workspace(
        compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
    ) -> bool:
        """Return whether enabled runtime capabilities require a workspace."""
        return FeatureFlags.is_flag_enabled(
            _MANAGED_WORKSPACE_HYDRATION_FEATURE_FLAG,
            default=False,
        ) and requires_managed_workspace(compiled_graph)

    def _select_deep_agent_hydration_policy(self) -> HydrationPolicy:
        """Select the UiPath-owned hydration lifecycle for this execution mode."""
        return HydrationPolicy.SUSPEND_OR_SUCCESS

    async def _create_managed_workspace(
        self,
        runtime_id: str,
    ) -> Workspace:
        """Create a clean, path-safe workspace for one runtime instance."""
        base_dir = (
            Path(self.context.runtime_dir or "__uipath") / "workspaces"
        ).resolve()
        workspace_id = hashlib.sha256(runtime_id.encode("utf-8")).hexdigest()
        workspace_path = base_dir / workspace_id
        if workspace_path.exists():
            await asyncio.to_thread(shutil.rmtree, workspace_path)
        return Workspace.create(workspace_path, cleanup=True)

    async def new_runtime(
        self,
        entrypoint: str,
        runtime_id: str,
        **kwargs,
    ) -> UiPathRuntimeProtocol:
        """
        Create a new LangGraph runtime instance.

        Args:
            entrypoint: Graph name from langgraph.json
            runtime_id: Unique identifier for the runtime instance
            **kwargs: Forwarded factory kwargs. Recognized:
                ``evaluator`` (``EvaluatorProtocol``) — when present, the
                factory wires a :class:`GovernanceCallbackHandler` into
                the runtime's callback list.

        Returns:
            Configured runtime instance with compiled graph
        """
        # Get shared memory instance
        memory = await self._get_memory()

        compiled_graph = await self._resolve_and_compile_graph(
            entrypoint, memory, **kwargs
        )

        return await self._create_runtime_instance(
            compiled_graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            **kwargs,
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
