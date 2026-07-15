from types import SimpleNamespace

from uipath_langchain.deepagents.metadata import (
    is_deep_agent_graph,
)


def test_deepagents_langsmith_metadata_is_detected() -> None:
    graph = SimpleNamespace(config={"metadata": {"ls_integration": "deepagents"}})

    assert is_deep_agent_graph(graph)


def test_deepagents_langsmith_metadata_is_detected_on_subgraph_node() -> None:
    subgraph = SimpleNamespace(config={"metadata": {"ls_integration": "deepagents"}})
    graph = SimpleNamespace(
        nodes={"advanced_agent": SimpleNamespace(runnable=subgraph)}
    )

    assert is_deep_agent_graph(graph)
