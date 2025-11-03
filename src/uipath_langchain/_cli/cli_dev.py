import asyncio
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._dev._terminal import UiPathDevTerminal
from uipath._cli._runtime._contracts import UiPathRuntimeContext, UiPathRuntimeFactory
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli.middlewares import MiddlewareResult

from .._tracing import _instrument_traceable_attributes
from ._runtime._memory import get_memory
from ._runtime._runtime import LangGraphScriptRuntime
from ._utils._graph import LangGraphConfig

console = ConsoleLogger()


def langgraph_dev_middleware(interface: Optional[str]) -> MiddlewareResult:
    """Middleware to launch the developer terminal"""
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        if interface == "terminal":
            _instrument_traceable_attributes()

            async def execute():
                context = UiPathRuntimeContext.with_defaults()

                async with get_memory(context) as memory:
                    runtime_factory = UiPathRuntimeFactory(
                        LangGraphScriptRuntime,
                        UiPathRuntimeContext,
                        lambda ctx: LangGraphScriptRuntime(ctx, memory, ctx.entrypoint),
                    )

                    runtime_factory.add_instrumentor(
                        LangChainInstrumentor, get_current_span
                    )

                    app = UiPathDevTerminal(runtime_factory)

                    await app.run_async()

            asyncio.run(execute())
        else:
            console.error(f"Unknown interface: {interface}")
    except KeyboardInterrupt:
        console.info("Debug session interrupted by user")
    except Exception as e:
        console.error(f"Error occurred: {e}")
        return MiddlewareResult(
            should_continue=False,
            should_include_stacktrace=True,
        )

    return MiddlewareResult(should_continue=False)
