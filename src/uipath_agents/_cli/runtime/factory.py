from dotenv import load_dotenv
from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeProtocol
from uipath_langchain._cli._runtime._factory import LangGraphRuntimeFactory

from ..._observability import configure_telemetry, shutdown_telemetry
from ..constants import AGENT_ENTRYPOINT
from .runtime import create_agent_langgraph_runtime

load_dotenv()


class AgentRuntimeFactory(LangGraphRuntimeFactory):
    """Factory for creating Agent runtimes from agent.json configuration."""

    def __init__(self, context: UiPathRuntimeContext) -> None:
        """Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
        super().__init__(context)
        configure_telemetry(context.trace_manager)

    def discover_entrypoints(self) -> list[str]:
        """Discover the Agent entrypoint agent.json"""
        return [AGENT_ENTRYPOINT]

    async def discover_runtimes(self) -> list[UiPathRuntimeProtocol]:
        """Discover runtime instances for all entrypoints."""
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

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        if self.context.trace_manager:
            self.context.trace_manager.flush_spans()
        shutdown_telemetry()
        await super().dispose()
