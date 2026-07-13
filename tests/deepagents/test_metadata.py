from unittest.mock import patch

import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from uipath_langchain.deepagents import (
    UiPathDeepAgentRuntimeSpec,
    get_uipath_deep_agent_runtime_spec,
    set_uipath_deep_agent_runtime_spec,
)
from uipath_langchain.deepagents.agent import create_uipath_deep_agent_graph


class State(BaseModel):
    value: str = ""


class Output(BaseModel):
    result: str


def _graph() -> StateGraph:
    async def node(state: State) -> State:
        return state

    builder = StateGraph(State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_edge("node", END)
    return builder


def test_runtime_spec_round_trips_on_builder_and_compiled_graph() -> None:
    graph = _graph()
    spec = UiPathDeepAgentRuntimeSpec(workspace_config_key="workspace")

    set_uipath_deep_agent_runtime_spec(graph, spec)
    compiled = graph.compile()

    assert get_uipath_deep_agent_runtime_spec(graph) == spec
    assert get_uipath_deep_agent_runtime_spec(compiled) == spec


def test_runtime_spec_defaults_to_success_persistence() -> None:
    spec = UiPathDeepAgentRuntimeSpec()

    assert spec.hydration_policy == "suspend_or_success"


def test_uipath_deep_agent_graph_exposes_hydration_policy() -> None:
    with patch(
        "uipath_langchain.deepagents.agent.create_advanced_agent_graph",
        return_value=_graph(),
    ):
        graph = create_uipath_deep_agent_graph(
            model=object(),  # type: ignore[arg-type]
            output_schema=Output,
            hydration_policy="always",
        )

    spec = get_uipath_deep_agent_runtime_spec(graph)
    assert spec is not None
    assert spec.hydration_policy == "always"


def test_conversational_mode_is_explicitly_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Conversational UiPath DeepAgents"):
        create_uipath_deep_agent_graph(
            model=object(),  # type: ignore[arg-type]
            output_schema=Output,
            interaction_mode="conversation",
        )


def test_subagents_are_forwarded_to_advanced_agent_graph() -> None:
    subagent = {
        "name": "researcher",
        "description": "Researches one part of the task",
        "system_prompt": "Research carefully.",
    }

    with patch(
        "uipath_langchain.deepagents.agent.create_advanced_agent_graph",
        return_value=_graph(),
    ) as mock_create:
        create_uipath_deep_agent_graph(
            model=object(),  # type: ignore[arg-type]
            output_schema=Output,
            subagents=[subagent],
        )

    assert mock_create.call_args.kwargs["subagents"] == [subagent]
