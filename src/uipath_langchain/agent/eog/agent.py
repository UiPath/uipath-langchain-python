"""EoG (Explanations over Graphs) agent graph builder."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from .nodes import (
    _make_bootstrap,
    _make_fetch_context,
    _make_policy,
    frontier_node,
    pop_node,
    propagate_node,
    should_continue,
    update_node,
)
from .ontology_client import OntologyClient
from .types import EoGState, InvestigationConfig


def create_eog_agent(
    model: BaseChatModel,
    ontology_client: OntologyClient,
    ontology_name: str,
    *,
    investigation_config: InvestigationConfig | None = None,
) -> StateGraph:
    """Create an EoG investigation agent as a LangGraph StateGraph.

    The graph performs belief-propagation over an ontology: it seeds beliefs
    for a set of starting entities, visits each entity via BFS, uses an LLM
    to assign a label, and propagates updates to neighbours until the budget
    is exhausted or the active set is empty.

    Args:
        model: LLM for the abductive policy node.
        ontology_client: REST client for the ontology-runtime.
        ontology_name: Name/ID of the deployed ontology.
        investigation_config: Configuration for the investigation
            (labels, seeds, budget).

    Returns:
        Uncompiled ``StateGraph``. Call ``.compile()`` to get a runnable
        graph.
    """
    builder: StateGraph = StateGraph(EoGState)

    builder.add_node(
        "bootstrap",
        _make_bootstrap(ontology_client, ontology_name, investigation_config),
    )
    builder.add_node("pop", pop_node)
    builder.add_node(
        "fetch_context",
        _make_fetch_context(ontology_client, ontology_name),
    )
    builder.add_node("policy", _make_policy(model))
    builder.add_node("update", update_node)
    builder.add_node("propagate", propagate_node)
    builder.add_node("frontier", frontier_node)

    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "pop")
    builder.add_conditional_edges(
        "pop",
        should_continue,
        {"fetch_context": "fetch_context", "frontier": "frontier"},
    )
    builder.add_edge("fetch_context", "policy")
    builder.add_edge("policy", "update")
    builder.add_edge("update", "propagate")
    builder.add_edge("propagate", "pop")
    builder.add_edge("frontier", END)

    return builder
