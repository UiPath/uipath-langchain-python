import os
from pathlib import Path
from typing import Callable, Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._runtime._contracts import UiPathRuntimeFactory
from uipath_langchain._cli._runtime._context import LangGraphRuntimeContext
from uipath_langchain._cli._runtime._runtime import (
    LangGraphRuntime,
    LangGraphScriptRuntime,
)
from uipath_langchain._tracing import _instrument_traceable_attributes

from ..agent_graph_builder import build_agent_graph
from .constants import AGENT_FILENAME
from .agent_loader import load_agent_configuration
from .json_schema_utils import validate_json_against_json_schema


class AgentLangGraphRuntime(LangGraphRuntime):
    pass


def create_graph_builder(
    input_data: Optional[str] = None, resume: bool = False
) -> Callable:
    """
    Create a graph builder function that loads the agent configuration
    and builds the agent graph.

    Args:
        input_data: Optional input data to pass to the graph builder
        resume: Whether this is a resume operation (skips input validation)

    Returns:
        Async callable that builds and returns the agent graph
    """

    async def graph_builder():
        agent_json_path = Path.cwd() / AGENT_FILENAME
        agent_definition = load_agent_configuration(agent_json_path)

        validated_input_data = input_data
        if not resume:
            validated_input_data = validate_json_against_json_schema(
                agent_definition.input_schema, input_data
            )

        return await build_agent_graph(
            agent_definition, input_data=validated_input_data, feature_flags={}
        )

    return graph_builder


def create_agent_langgraph_runtime(ctx: LangGraphRuntimeContext) -> LangGraphRuntime:
    """
    Create an AgentLangGraphRuntime instance with state file cleanup logic.

    Args:
        ctx: Runtime context containing execution parameters

    Returns:
        Configured AgentLangGraphRuntime instance
    """
    graph_builder = create_graph_builder(ctx.input, resume=ctx.resume)
    runtime = AgentLangGraphRuntime(ctx, graph_builder)

    # If not resuming and no job id, delete the previous state file
    if not ctx.resume and ctx.job_id is None:
        if os.path.exists(runtime.state_file_path):
            os.remove(runtime.state_file_path)

    return runtime


def setup_runtime_factory(
    runtime_generator: Optional[
        Callable[[LangGraphRuntimeContext], LangGraphRuntime]
    ] = None,
    context_generator: Optional[Callable[[], LangGraphRuntimeContext]] = None,
) -> UiPathRuntimeFactory:
    """
    Set up a UiPathRuntimeFactory with common configuration for LangGraph agents.

    This configures:
    - LangChain instrumentation for tracing
    - Traceable attributes for observability

    Args:
        runtime_generator: Optional custom runtime generator function.
                          If not provided, uses create_runtime
        context_generator: Optional context generator function

    Returns:
        Configured UiPathRuntimeFactory instance
    """
    # Use default runtime generator if not provided
    if runtime_generator is None:
        runtime_generator = create_agent_langgraph_runtime

    # Set up tracing instrumentation
    _instrument_traceable_attributes()

    # Create runtime factory
    factory_kwargs = {
        "runtime_type": LangGraphScriptRuntime,
        "context_type": LangGraphRuntimeContext,
        "runtime_generator": runtime_generator,
    }

    if context_generator is not None:
        factory_kwargs["context_generator"] = context_generator

    runtime_factory = UiPathRuntimeFactory(**factory_kwargs)
    runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)

    return runtime_factory
