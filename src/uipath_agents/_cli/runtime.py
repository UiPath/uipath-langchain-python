import logging
import os
from pathlib import Path
from typing import Callable, Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from typing_extensions import Any
from uipath._cli._runtime._contracts import UiPathRuntimeContext, UiPathRuntimeFactory
from uipath_langchain._cli._runtime._runtime import (
    LangGraphRuntime,
)
from uipath_langchain._tracing import _instrument_traceable_attributes

from ..agent_graph_builder import build_agent_graph
from .agent_loader import load_agent_configuration
from .constants import AGENT_FILENAME
from .json_schema_utils import validate_json_against_json_schema

logger = logging.getLogger(__name__)


class AgentLangGraphRuntime(LangGraphRuntime):
    """Low-code agent runtime extending LangGraph base runtime."""

    pass


def create_agent_langgraph_runtime(
    ctx: UiPathRuntimeContext, memory: AsyncSqliteSaver
) -> LangGraphRuntime:
    """Create runtime for low-code agents with input validation.

    Args:
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
        if not ctx.resume:
            agent_input = validate_json_against_json_schema(
                agent_definition.input_schema, ctx.input
            )

        return await build_agent_graph(agent_definition, input_data=agent_input)

    runtime = AgentLangGraphRuntime(ctx, graph_builder, memory)

    return runtime


def setup_runtime_factory(
    runtime_generator: Callable[[UiPathRuntimeContext], LangGraphRuntime],
    context_generator: Optional[Callable[[], UiPathRuntimeContext]] = None,
) -> UiPathRuntimeFactory[LangGraphRuntime, UiPathRuntimeContext]:
    """Set up runtime factory with instrumentation for low-code agents."""

    os.environ.setdefault("OTEL_SERVICE_NAME", "uipath-agents")

    _instrument_traceable_attributes()

    runtime_factory = UiPathRuntimeFactory(
        LangGraphRuntime,
        UiPathRuntimeContext,
        runtime_generator=runtime_generator,
        context_generator=context_generator,
    )

    runtime_factory.add_instrumentor(AsyncioInstrumentor, lambda: None)
    runtime_factory.add_instrumentor(HTTPXClientInstrumentor, lambda: None)
    runtime_factory.add_instrumentor(AioHttpClientInstrumentor, lambda: None)
    runtime_factory.add_instrumentor(SQLite3Instrumentor, lambda: None)
    runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)

    return runtime_factory
