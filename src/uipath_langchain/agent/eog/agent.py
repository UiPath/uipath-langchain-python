"""EoG (Explanations over Graphs) agent graph builder.

Lazy traversal — no upfront graph fetch. Function definitions are the
navigation contract. The agent discovers the graph as it walks.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from .nodes import (
    _make_discover,
    _make_gather,
    _make_label,
    _make_seed,
    pop_node,
    propagate_node,
    should_continue,
    synthesize_node,
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

    The graph discovers the ontology lazily via function definitions:
    each entity type's functions are fetched on first visit (cached),
    and records are discovered from function results.

    Args:
        model: LLM for the abductive label node.
        ontology_client: REST client for the ontology-runtime.
        ontology_name: Name/ID of the deployed ontology.
        investigation_config: Configuration for the investigation
            (labels, seeds, budget).

    Returns:
        Uncompiled ``StateGraph``. Call ``.compile()`` to get a runnable.
    """
    builder: StateGraph = StateGraph(EoGState)

    builder.add_node("seed", _make_seed(investigation_config))
    builder.add_node("pop", pop_node)
    builder.add_node(
        "discover", _make_discover(ontology_client, ontology_name),
    )
    builder.add_node(
        "gather", _make_gather(ontology_client, ontology_name),
    )
    builder.add_node("label", _make_label(model))
    builder.add_node("update", update_node)
    builder.add_node("propagate", propagate_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "seed")
    builder.add_edge("seed", "pop")
    builder.add_conditional_edges(
        "pop",
        should_continue,
        {"discover": "discover", "synthesize": "synthesize"},
    )
    builder.add_edge("discover", "gather")
    builder.add_edge("gather", "label")
    builder.add_edge("label", "update")
    builder.add_edge("update", "propagate")
    builder.add_edge("propagate", "pop")
    builder.add_edge("synthesize", END)

    return builder
