from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from .._cli._runtime._factory import LangGraphRuntimeFactory


def register_runtime_factory() -> None:
    """Register the LangGraph factory. Called automatically via entry point."""

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return LangGraphRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register("langgraph", create_factory, "langgraph.json")


register_runtime_factory()

__all__ = ["LangGraphRuntimeFactory", "register_runtime_factory"]
