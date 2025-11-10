import asyncio
from typing import Optional

from uipath._cli._dev._terminal import UiPathDevTerminal
from uipath._cli._runtime._contracts import UiPathRuntimeContext
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli.middlewares import MiddlewareResult
from uipath_langchain._cli._runtime._memory import get_memory

from .runtime import create_agent_langgraph_runtime, setup_runtime_factory

console = ConsoleLogger()


def lowcode_dev_middleware(interface: Optional[str]) -> MiddlewareResult:
    """Middleware to launch the developer terminal"""

    try:
        if interface == "terminal":

            async def execute():
                context = UiPathRuntimeContext.with_defaults()

                async with get_memory(context) as memory:
                    runtime_factory = setup_runtime_factory(
                        runtime_generator=lambda ctx: create_agent_langgraph_runtime(
                            ctx, memory
                        )
                    )

                    app = UiPathDevTerminal(runtime_factory)

                    await app.run_async()

            asyncio.run(execute())
        else:
            console.error(f"Unknown interface: {interface}")
            return MiddlewareResult(
                should_continue=False, error_message=f"Unknown interface: {interface}"
            )
    except KeyboardInterrupt:
        console.info("Debug session interrupted by user")
    except Exception as e:
        console.error(f"Error occurred: {e}")
        return MiddlewareResult(
            should_continue=False,
            should_include_stacktrace=True,
        )

    return MiddlewareResult(should_continue=False)
