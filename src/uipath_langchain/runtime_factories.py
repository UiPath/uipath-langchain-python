"""Runtime factory for LangGraph projects."""

from uipath._cli._runtime._contracts import UiPathRuntimeFactory

from ._cli._runtime._context import LangGraphRuntimeContext
from ._cli._runtime._runtime import LangGraphScriptRuntime


class LangGraphRuntimeFactory(
    UiPathRuntimeFactory[LangGraphScriptRuntime, LangGraphRuntimeContext]
):
    """Factory for LangGraph runtimes."""

    def __init__(self):
        super().__init__(
            LangGraphScriptRuntime,
            LangGraphRuntimeContext,
            context_generator=lambda **kwargs: LangGraphRuntimeContext.with_defaults(
                **kwargs
            ),
        )
