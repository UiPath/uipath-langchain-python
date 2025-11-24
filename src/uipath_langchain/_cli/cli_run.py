import asyncio
import os
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._debug._bridge import ConsoleDebugBridge, UiPathDebugBridge
from uipath._cli._runtime._contracts import (
    UiPathRuntimeContext,
    UiPathRuntimeFactory,
    UiPathRuntimeResult,
)
from uipath._cli.middlewares import MiddlewareResult
from uipath._cli.cli_run import MemorySpanExporter, _generate_evaluation_set
from uipath._cli._utils._console import ConsoleLogger
from uipath._events._events import UiPathAgentStateEvent
from uipath.tracing import JsonLinesFileExporter, LlmOpsHttpExporter

from .._tracing import (
    _instrument_traceable_attributes,
)

# Create console instance
console = ConsoleLogger()
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._memory import get_memory
from ._runtime._runtime import LangGraphScriptRuntime
from ._utils._graph import LangGraphConfig


def langgraph_run_middleware(
    entrypoint: Optional[str],
    input: Optional[str],
    resume: bool,
    trace_file: Optional[str] = None,
    **kwargs,
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""
    # Extract evaluation parameters from kwargs (passed by the CLI)
    generate_evals = kwargs.get("generate_evals")
    eval_evaluators = kwargs.get("eval_evaluators")
    input_file = kwargs.get("input_file")
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        _instrument_traceable_attributes()

        execution_result = None
        memory_span_exporter = None

        async def execute():
            nonlocal execution_result, memory_span_exporter
            context = UiPathRuntimeContext.with_defaults(**kwargs)
            context.entrypoint = entrypoint
            context.input = input
            context.resume = resume

            async with get_memory(context) as memory:
                runtime_factory = UiPathRuntimeFactory(
                    LangGraphScriptRuntime,
                    UiPathRuntimeContext,
                    runtime_generator=lambda ctx: LangGraphScriptRuntime(
                        ctx, memory, ctx.entrypoint
                    ),
                )

                runtime_factory.add_instrumentor(
                    LangChainInstrumentor, get_current_span
                )

                if trace_file:
                    runtime_factory.add_span_exporter(JsonLinesFileExporter(trace_file))

                # Add memory span exporter if generating evals to capture node-level data
                if generate_evals:
                    memory_span_exporter = MemorySpanExporter()
                    runtime_factory.add_span_exporter(memory_span_exporter, batch=False)

                if context.job_id:
                    runtime_factory.add_span_exporter(
                        LlmOpsHttpExporter(extra_process_spans=True)
                    )
                    execution_result = await runtime_factory.execute(context)
                else:
                    debug_bridge: UiPathDebugBridge = ConsoleDebugBridge()
                    await debug_bridge.emit_execution_started("default")
                    async for event in runtime_factory.stream(context):
                        if isinstance(event, UiPathRuntimeResult):
                            execution_result = event
                            await debug_bridge.emit_execution_completed(event)
                        elif isinstance(event, UiPathAgentStateEvent):
                            await debug_bridge.emit_state_update(event)

        asyncio.run(execute())

        # Generate evaluation set if requested
        if generate_evals and execution_result:
            # Check if execution was interrupted (HITL) - skip eval generation for interrupted runs
            output_for_eval = execution_result.output if hasattr(execution_result, 'output') else execution_result

            # Check if output contains an Interrupt object (from langgraph.types.interrupt)
            is_interrupted = False
            if isinstance(output_for_eval, dict):
                # Check for __interrupt__ key which indicates HITL interruption
                is_interrupted = '__interrupt__' in output_for_eval

            if is_interrupted:
                console.info("Execution was interrupted (HITL). Skipping evaluation generation for interrupted runs.")
            else:
                # Get the actual input data (from file or argument)
                actual_input = input
                if input_file and os.path.exists(input_file):
                    try:
                        with open(input_file, 'r') as f:
                            actual_input = f.read()
                    except Exception as e:
                        console.warning(f"Failed to read input file for eval generation: {e}")

                # If output is a Pydantic model, convert to dict
                if hasattr(output_for_eval, 'model_dump'):
                    output_for_eval = output_for_eval.model_dump()
                elif hasattr(output_for_eval, 'dict'):
                    output_for_eval = output_for_eval.dict()

                # Get spans from memory exporter if available
                collected_spans = memory_span_exporter.spans if memory_span_exporter else None

                # Extract LangGraph node names from the compiled graph for filtering
                import sys
                langgraph_nodes = None
                try:
                    # Try to access the already-loaded graph from sys.modules

                    # Get the module name from the entrypoint path
                    module_dir = os.path.dirname(os.path.abspath(entrypoint))
                    module_name = os.path.splitext(os.path.basename(entrypoint))[0]

                    # Look for the module in sys.modules (it should already be loaded by the runtime)
                    # Try common patterns: 'main', 'agent', or the actual module name
                    possible_modules = [module_name, 'main', 'agent', '__main__']

                    module = None
                    for mod_name in possible_modules:
                        if mod_name in sys.modules:
                            module = sys.modules[mod_name]
                            break

                    if module and hasattr(module, 'graph'):
                        graph = module.graph
                        # Get node names from the compiled graph
                        if hasattr(graph, 'nodes'):
                            langgraph_nodes = list(graph.nodes.keys())
                        elif hasattr(graph, '_nodes'):
                            langgraph_nodes = list(graph._nodes.keys())

                        if langgraph_nodes:
                            console.info(f"Extracted {len(langgraph_nodes)} LangGraph nodes for filtering: {', '.join(langgraph_nodes)}")
                except Exception as e:
                    # Silently fail - we'll use default filtering
                    pass

                _generate_evaluation_set(
                    input_data=actual_input,
                    output_data=output_for_eval,
                    entrypoint=entrypoint,
                    eval_set_path=generate_evals,
                    evaluators=list(eval_evaluators) if eval_evaluators else None,
                    spans=collected_spans,
                    langgraph_nodes=langgraph_nodes,
                )

        return MiddlewareResult(
            should_continue=False,
            error_message=None,
        )

    except LangGraphRuntimeError as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=e.error_info.detail,
            should_include_stacktrace=True,
        )
    except Exception as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=f"Error: {str(e)}",
            should_include_stacktrace=True,
        )
