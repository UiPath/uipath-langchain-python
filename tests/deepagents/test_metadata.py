from types import SimpleNamespace
from unittest.mock import MagicMock

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel

from uipath_langchain.deepagents.metadata import (
    is_deep_agent_graph,
)


def test_standard_deepagent_is_detected_from_native_metadata() -> None:
    model = MagicMock(spec=BaseChatModel)
    model.profile = None

    graph = create_deep_agent(model=model)

    assert is_deep_agent_graph(graph)


def test_deepagents_langsmith_metadata_is_detected() -> None:
    graph = SimpleNamespace(config={"metadata": {"ls_integration": "deepagents"}})

    assert is_deep_agent_graph(graph)


def test_deepagents_langsmith_metadata_is_detected_on_subgraph_node() -> None:
    subgraph = SimpleNamespace(config={"metadata": {"ls_integration": "deepagents"}})
    graph = SimpleNamespace(
        nodes={"advanced_agent": SimpleNamespace(runnable=subgraph)}
    )

    assert is_deep_agent_graph(graph)
