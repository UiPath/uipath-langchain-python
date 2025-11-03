"""Runtime factory for LangGraph projects."""

from uipath._cli._runtime._contracts import UiPathRuntimeContext, UiPathRuntimeFactory

from ._cli._runtime._runtime import LangGraphScriptRuntime


class LangGraphRuntimeFactory(
    UiPathRuntimeFactory[LangGraphScriptRuntime, UiPathRuntimeContext]
):
    """Factory for LangGraph runtimes."""

    def __init__(self):
        super().__init__(
            LangGraphScriptRuntime,
            UiPathRuntimeContext,
            context_generator=lambda **kwargs: UiPathRuntimeContext.with_defaults(
                **kwargs
            ),
        )
