from uipath.runtime import (
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeFactoryRegistry,
)

from uipath_langchain.runtime._msgpack_allowlist import (
    register_uipath_msgpack_safe_types,
)
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime
from uipath_langchain.runtime.schema import (
    get_entrypoints_schema,
    get_graph_schema,
)


def register_runtime_factory() -> None:
    """Register the LangGraph factory. Called automatically via entry point."""
    # Allow UiPath HITL interrupt/recipient payloads to (de)serialize cleanly
    # through the LangGraph checkpoint (silences the "unregistered type" warning
    # and future-proofs against strict msgpack).
    register_uipath_msgpack_safe_types()

    def create_factory(
        context: UiPathRuntimeContext | None = None,
    ) -> UiPathRuntimeFactoryProtocol:
        return UiPathLangGraphRuntimeFactory(
            context=context if context else UiPathRuntimeContext(),
        )

    UiPathRuntimeFactoryRegistry.register("langgraph", create_factory, "langgraph.json")


register_runtime_factory()

__all__ = [
    "register_runtime_factory",
    "get_entrypoints_schema",
    "get_graph_schema",
    "UiPathLangGraphRuntimeFactory",
    "UiPathLangGraphRuntime",
]
