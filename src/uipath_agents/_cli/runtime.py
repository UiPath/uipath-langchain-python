import logging
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing_extensions import Any
from uipath.runtime import UiPathRuntimeContext
from uipath_langchain._cli._runtime._runtime import (
    LangGraphRuntime,
)

from ..agent_graph_builder import build_agent_graph
from .constants import AGENT_FILENAME
from .json_schema_utils import validate_json_against_json_schema
from .utils import load_agent_configuration

logger = logging.getLogger(__name__)


class AgentLangGraphRuntime(LangGraphRuntime):
    """Agents runtime extending LangGraph base runtime."""

    pass


def create_agent_langgraph_runtime(
    runtime_id: str, ctx: UiPathRuntimeContext, memory: AsyncSqliteSaver
) -> AgentLangGraphRuntime:
    """Create runtime for low-code agents with input validation.

    Args:
        runtime_id: Unique identifier for the runtime instance
        ctx: Runtime context containing input, resume flag, and other metadata
        memory: AsyncSqliteSaver instance for checkpoint/state management

    Returns:
        LangGraphRuntime instance configured for the low-code agent
    """

    async def graph_builder():
        """Load agent config and validate input (on new run) and build graph."""
        agent_json_path = Path.cwd() / AGENT_FILENAME
        agent_definition = load_agent_configuration(agent_json_path)

        agent_input: dict[str, Any] = {}
        # TODO: think if this needs to be moved elsewhere??
        if not ctx.resume:
            agent_input = validate_json_against_json_schema(
                agent_definition.input_schema, ctx.input
            )

        return await build_agent_graph(agent_definition, input_data=agent_input)

    runtime = AgentLangGraphRuntime(runtime_id, graph_builder, memory)

    return runtime
