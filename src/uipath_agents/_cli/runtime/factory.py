from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph, StateGraph
from uipath.core import UiPathTraceManager
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain.runtime.errors import LangGraphErrorCode, LangGraphRuntimeError
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory
from uipath_langchain.runtime.storage import SqliteResumableStorage

from uipath_agents.agent_graph_builder import build_agent_graph

from ..._observability import configure_telemetry, shutdown_telemetry
from ..constants import AGENT_ENTRYPOINT
from ..utils import _prepare_agent_run_files, load_agent_configuration
from .runtime import AgentsLangGraphRuntime
from .utils import validate_json_against_json_schema

load_dotenv()


class AgentsRuntimeFactory(UiPathLangGraphRuntimeFactory):
    """Factory for creating Agent runtimes from agent.json configuration."""

    def __init__(self, context: UiPathRuntimeContext) -> None:
        """Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
        super().__init__(context)

    def _setup_instrumentation(self, trace_manager: UiPathTraceManager | None) -> None:
        """Setup tracing and instrumentation."""
        super()._setup_instrumentation(trace_manager)
        configure_telemetry(trace_manager)

    def discover_entrypoints(self) -> list[str]:
        """Discover the Agent entrypoint agent.json"""
        return [AGENT_ENTRYPOINT]

    async def _load_graph(
        self, entrypoint: str
    ) -> StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any]:
        """Load agent graph for the given entrypoint.

        Args:
            entrypoint: Agent file path (agent.json)

        Returns:
            Compiled StateGraph for the agent

        Raises:
            LangGraphRuntimeError: If graph cannot be loaded
        """
        try:
            _prepare_agent_run_files()

            agent_json_path = Path.cwd() / entrypoint
            agent_definition = load_agent_configuration(agent_json_path)

            agent_input: dict[str, Any] = {}
            if not self.context.resume:
                agent_input = validate_json_against_json_schema(
                    agent_definition.input_schema, self.context.input
                )

            return await build_agent_graph(agent_definition, input_data=agent_input)

        except FileNotFoundError as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_NOT_FOUND,
                "Agent configuration not found",
                f"Agent file '{entrypoint}' not found: {str(e)}",
                UiPathErrorCategory.DEPLOYMENT,
            ) from e
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_LOAD_ERROR,
                "Failed to load agent graph",
                f"Unexpected error loading agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    async def _create_runtime_instance(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
        runtime_id: str,
        entrypoint: str,
    ) -> UiPathRuntimeProtocol:
        """Create an agent runtime instance from a compiled graph.

        Args:
            compiled_graph: The compiled graph
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name

        Returns:
            Configured AgentLangGraphRuntime instance wrapped in UiPathResumableRuntime

        Note:
            We override this to use AgentLangGraphRuntime instead of UiPathLangGraphRuntime.
            The parent method hardcodes the runtime class, so we can't call super() here.
        """
        base_runtime = AgentsLangGraphRuntime(
            graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
        )

        return await self._wrap_in_resumable_runtime(base_runtime)

    async def _wrap_in_resumable_runtime(
        self, base_runtime: UiPathRuntimeProtocol
    ) -> UiPathResumableRuntime:
        """Wrap a base runtime in UiPathResumableRuntime with storage and trigger manager.

        Args:
            base_runtime: The base runtime to wrap

        Returns:
            UiPathResumableRuntime wrapping the base runtime
        """
        memory = await self._get_memory()
        storage = SqliteResumableStorage(memory)
        trigger_manager = UiPathResumeTriggerHandler()

        return UiPathResumableRuntime(
            delegate=base_runtime,
            storage=storage,
            trigger_manager=trigger_manager,
        )

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        if self.context.trace_manager:
            self.context.trace_manager.flush_spans()
        shutdown_telemetry()
        await super().dispose()
