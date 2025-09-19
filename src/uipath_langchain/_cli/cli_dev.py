import asyncio
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._dev._terminal import UiPathDevTerminal
from uipath._cli._runtime._contracts import UiPathRuntimeFactory
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli.middlewares import MiddlewareResult

from .._tracing import _instrument_traceable_attributes
from ._runtime._context import LangGraphRuntimeContext
from ._runtime._runtime import LangGraphRuntime

console = ConsoleLogger()


def langgraph_dev_middleware(interface: Optional[str]) -> MiddlewareResult:
    """Middleware to launch the developer terminal"""

    try:
        if interface == "terminal":
            runtime_factory = UiPathRuntimeFactory(
                LangGraphRuntime, LangGraphRuntimeContext
            )

            _instrument_traceable_attributes()
            runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)
            app = UiPathDevTerminal(runtime_factory)
            asyncio.run(app.run_async())
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
