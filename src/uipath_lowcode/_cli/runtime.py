import logging
import os
from pathlib import Path
from typing import Callable, Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from uipath._cli._runtime._contracts import UiPathRuntimeFactory
from uipath_langchain._cli._runtime._context import LangGraphRuntimeContext
from uipath_langchain._cli._runtime._runtime import (
    LangGraphRuntime,
    LangGraphScriptRuntime,
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


def create_agent_langgraph_runtime(ctx: LangGraphRuntimeContext) -> LangGraphRuntime:
    """Create runtime for low-code agents with input validation and state cleanup."""

    async def graph_builder():
        """Load agent config and validate input (on new run) and build graph."""
        agent_json_path = Path.cwd() / AGENT_FILENAME
        agent_definition = load_agent_configuration(agent_json_path)

        agent_input = ctx.input
        if not ctx.resume:
            agent_input = validate_json_against_json_schema(
                agent_definition.input_schema, ctx.input
            )

        return await build_agent_graph(agent_definition, input_data=agent_input)

    if not ctx.resume:
        if ctx.runtime_dir and ctx.state_file:
            state_path = os.path.join(ctx.runtime_dir, ctx.state_file)
            try:
                os.remove(state_path)
                logger.debug(f"Deleted old state file: {state_path}")
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning(f"Could not delete state file {state_path}: {e}")

    runtime = AgentLangGraphRuntime(ctx, graph_builder)

    return runtime


def setup_runtime_factory(
    runtime_generator: Optional[
        Callable[[LangGraphRuntimeContext], LangGraphRuntime]
    ] = None,
    context_generator: Optional[Callable[[], LangGraphRuntimeContext]] = None,
) -> UiPathRuntimeFactory:
    """Set up runtime factory with instrumentation for low-code agents."""
    if runtime_generator is None:
        runtime_generator = create_agent_langgraph_runtime

    os.environ.setdefault("OTEL_SERVICE_NAME", "uipath-lowcode-python")
    os.environ.setdefault("OTEL_SERVICE_VERSION", "0.0.1")

    _instrument_traceable_attributes()

    runtime_factory = UiPathRuntimeFactory(
        runtime_class=LangGraphScriptRuntime,
        context_class=LangGraphRuntimeContext,
        runtime_generator=runtime_generator,
        context_generator=context_generator,
    )

    runtime_factory.add_instrumentor(AsyncioInstrumentor, lambda: None)
    runtime_factory.add_instrumentor(HTTPXClientInstrumentor, lambda: None)
    runtime_factory.add_instrumentor(AioHttpClientInstrumentor, lambda: None)
    runtime_factory.add_instrumentor(SQLite3Instrumentor, lambda: None)
    runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)

    return runtime_factory
