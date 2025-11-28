import logging
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from uipath.runtime import (
    UiPathRuntimeContext,
)
from uipath.runtime.schema import UiPathRuntimeSchema
from uipath_langchain._cli._runtime._runtime import (
    LangGraphRuntime,
)

from uipath_agents.agent_graph_builder import build_agent_graph

from ..utils import load_agent_configuration
from .utils import validate_json_against_json_schema

logger = logging.getLogger(__name__)


class AgentLangGraphRuntime(LangGraphRuntime):
    """Agents runtime extending LangGraph base runtime."""

    def __init__(
        self,
        runtime_id: str,
        graph_resolver: Any,
        memory: AsyncSqliteSaver,
        entrypoint: str,
    ) -> None:
        super().__init__(runtime_id, graph_resolver, memory)
        self.entrypoint = entrypoint

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Return the runtime schema for this agent."""
        agent_json_path = Path.cwd() / self.entrypoint
        agent_definition = load_agent_configuration(agent_json_path)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint,
            uniqueId=self.runtime_id,
            type="agent",
            input=agent_definition.input_schema or {},
            output=agent_definition.output_schema or {},
        )


def create_agent_langgraph_runtime(
    runtime_id: str,
    entrypoint: str,
    ctx: UiPathRuntimeContext,
    memory: AsyncSqliteSaver,
) -> AgentLangGraphRuntime:
    """Create runtime for agents with input validation.

    Args:
        runtime_id: Unique identifier for the runtime instance
        entrypoint: Agent file path containing the Agent definition
        ctx: Runtime context containing input, resume flag, and other metadata
        memory: AsyncSqliteSaver instance for checkpoint/state management

    Returns:
        AgentLangGraphRuntime instance configured for the agent
    """

    async def graph_resolver():
        """Load agent config, validate input on new runs, and build graph."""
        agent_json_path = Path.cwd() / entrypoint
        agent_definition = load_agent_configuration(agent_json_path)

        agent_input: dict[str, Any] = {}
        if not ctx.resume:
            agent_input = validate_json_against_json_schema(
                agent_definition.input_schema, ctx.input
            )

        return await build_agent_graph(agent_definition, input_data=agent_input)

    return AgentLangGraphRuntime(runtime_id, graph_resolver, memory, entrypoint)
