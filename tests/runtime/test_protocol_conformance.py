"""Protocol conformance for uipath-runtime protocol implementations.

Every protocol-annotated assignment below is a typed boundary: mypy verifies
the implementation against the protocol surface of the installed
uipath-runtime version. A dependency bump that adds or changes protocol
members fails typecheck on these lines until the implementations catch up.

The wiring mirrors UiPathLangGraphRuntimeFactory._create_runtime_instance,
so the exact production composition is what gets checked. Deliberately
construction-only: no runtime behavior is exercised here.
"""

from typing import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathResumableStorageProtocol,
    UiPathResumeTriggerProtocol,
    UiPathRuntimeContext,
    UiPathRuntimeFactoryProtocol,
    UiPathRuntimeProtocol,
)

from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime
from uipath_langchain.runtime.storage import SqliteResumableStorage


class _State(TypedDict, total=False):
    value: str


async def test_langgraph_wiring_satisfies_runtime_protocols() -> None:
    graph = StateGraph(_State)
    graph.add_node("noop", lambda state: state)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        compiled_graph = graph.compile(checkpointer=memory)

        delegate: UiPathRuntimeProtocol = UiPathLangGraphRuntime(
            graph=compiled_graph,
            runtime_id="protocol-conformance-test",
            entrypoint="test",
        )
        storage: UiPathResumableStorageProtocol = SqliteResumableStorage(memory)
        trigger_manager: UiPathResumeTriggerProtocol = UiPathResumeTriggerHandler()

        runtime: UiPathRuntimeProtocol = UiPathResumableRuntime(
            delegate=delegate,
            storage=storage,
            trigger_manager=trigger_manager,
            runtime_id="protocol-conformance-test",
        )

        assert runtime is not None


def test_langgraph_factory_satisfies_factory_protocol() -> None:
    factory: UiPathRuntimeFactoryProtocol = UiPathLangGraphRuntimeFactory(
        context=UiPathRuntimeContext()
    )

    assert factory is not None
