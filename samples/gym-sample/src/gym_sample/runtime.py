import importlib
from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import override
from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain.chat import UiPathAzureChatOpenAI
from uipath_langchain.runtime import UiPathLangGraphRuntimeFactory
from uipath_langchain.runtime.config import LangGraphConfig
from uipath_langchain.runtime.errors import LangGraphErrorCode, LangGraphRuntimeError

from gym_sample.uipath_gym_types import BasicLoop


class GymFactory(UiPathLangGraphRuntimeFactory):
    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.entrypaths = {}

    @override
    def _load_config(self) -> LangGraphConfig:
        if self._config is None:
            self._config = LangGraphConfig(config_path="gym.json")
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

        try:
            loop = self._get_agent_loop(config.graphs[entrypoint])
            # Build unified graph that accepts input at runtime
            return loop.build_graph()
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

    def _get_agent_loop(self, entrypath: str) -> BasicLoop:
        """Cached agent loop."""
        if entrypath not in self.entrypaths:
            module_path, agent_builder_name = entrypath.split(":")
            module = importlib.import_module(module_path)
            agent_builder = getattr(module, agent_builder_name)
            self.entrypaths[entrypath] = BasicLoop(
                scenario=agent_builder(),
                llm=UiPathAzureChatOpenAI(model="gpt-4o-2024-11-20"),
                print_trace=True,
                parallel_tool_calls=False,
            )
        return self.entrypaths[entrypath]


def register_runtime_factory() -> None:
    """Register the Gym factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return GymFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register("gym", create_factory, "gym.json")
