import inspect
from unittest.mock import MagicMock

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from uipath_langchain.deepagents import (
    create_uipath_deep_agent,
    with_uipath_managed_workspace,
)
from uipath_langchain.deepagents.metadata import requires_managed_workspace


class _State(TypedDict):
    value: str


def _model() -> BaseChatModel:
    model = MagicMock(spec=BaseChatModel)
    model.profile = None
    return model


def test_uipath_api_matches_upstream_authoring_parameters() -> None:
    upstream = inspect.signature(create_deep_agent).parameters
    uipath = inspect.signature(create_uipath_deep_agent).parameters
    runtime_owned = {
        "backend",
        "checkpointer",
        "store",
        "cache",
        "debug",
        "name",
        "interrupt_on",
    }

    assert list(uipath) == [name for name in upstream if name not in runtime_owned]
    for name, parameter in uipath.items():
        assert parameter.kind == upstream[name].kind
        assert parameter.default == upstream[name].default


def test_uipath_deepagent_has_explicit_runtime_metadata() -> None:
    graph = create_uipath_deep_agent(model=_model())

    assert requires_managed_workspace(graph)
    assert graph.config["metadata"]["uipath_runtime"] == {
        "requirements": ["managed_workspace"]
    }
    assert graph.config["metadata"]["ls_integration"] == "deepagents"


def test_standard_deepagent_has_no_managed_workspace_requirement() -> None:
    graph = create_deep_agent(model=_model())

    assert not requires_managed_workspace(graph)


def test_requirement_detection_tolerates_malformed_metadata() -> None:
    graph = MagicMock()
    graph.config = {"metadata": None}

    assert not requires_managed_workspace(graph)


def test_parent_requirement_is_explicitly_registered() -> None:
    child = create_uipath_deep_agent(model=_model())
    parent = StateGraph(_State)
    parent.add_node("deep_agent", child)
    parent.add_edge(START, "deep_agent")
    parent.add_edge("deep_agent", END)

    assert not requires_managed_workspace(parent.compile())
    assert requires_managed_workspace(with_uipath_managed_workspace(parent.compile()))
