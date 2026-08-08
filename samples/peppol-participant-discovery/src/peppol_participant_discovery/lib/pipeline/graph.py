"""LangGraph StateGraph construction for PEPPOL search pipeline."""

from __future__ import annotations

from typing import Literal, Optional

from langgraph.graph import END, StateGraph
from uipath import UiPath

from ..config import Settings
from ..models import CompanySignature
from .nodes import (
    create_refine_company_name_node,
    extract_payload_node,
    fetch_participant_details_node,
    finalize_results_node,
    peppol_search_naive_node,
    peppol_search_refined_node,
)
from .state import PipelineState


def should_fetch_details(state: PipelineState) -> Literal["fetch_details", "refine_name"]:
    """
    Conditional edge after naive search to decide next step.

    Returns
    -------
    Literal["fetch_details", "refine_name"]
        Route to details if found, or refinement if not found.
    """
    if state.get("naive_found"):
        return "fetch_details"
    else:
        return "refine_name"


def should_fetch_or_finalize(state: PipelineState) -> Literal["fetch_details", "finalize"]:
    """
    Conditional edge after refined search.

    Returns
    -------
    Literal["fetch_details", "finalize"]
        Route to details if found, or finalize if not found.
    """
    if state.get("refined_found"):
        return "fetch_details"
    else:
        return "finalize"


def build_pipeline_graph(
    uipath: Optional[UiPath],
    settings: Settings
) -> StateGraph:
    """
    Build the complete LangGraph StateGraph for PEPPOL search pipeline.

    Flow:
    1. Extract payload →
    2. PEPPOL search (naive) →
       - If found: fetch details → finalize
       - If not found: refine name → PEPPOL search (refined) →
         - If found: fetch details → finalize
         - If not found: finalize

    Parameters
    ----------
    uipath : Optional[UiPath]
        UiPath SDK instance (for LLM calls)
    settings : Settings
        Configuration settings

    Returns
    -------
    StateGraph
        Configured LangGraph StateGraph
    """
    # Create graph
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("extract_payload", extract_payload_node)
    graph.add_node("peppol_naive", peppol_search_naive_node)

    # Refine node needs UiPath SDK and settings injected
    refine_node = create_refine_company_name_node(uipath, settings)
    graph.add_node("refine_name", refine_node)

    graph.add_node("peppol_refined", peppol_search_refined_node)
    graph.add_node("fetch_details", fetch_participant_details_node)
    graph.add_node("finalize", finalize_results_node)

    # Define edges
    graph.set_entry_point("extract_payload")
    graph.add_edge("extract_payload", "peppol_naive")

    # After naive search: found? → fetch details, not found? → refine
    graph.add_conditional_edges(
        "peppol_naive",
        should_fetch_details,
        {
            "fetch_details": "fetch_details",
            "refine_name": "refine_name",
        },
    )

    # After refinement: always try refined search
    graph.add_edge("refine_name", "peppol_refined")

    # After refined search: found? → fetch details, not found? → finalize
    graph.add_conditional_edges(
        "peppol_refined",
        should_fetch_or_finalize,
        {
            "fetch_details": "fetch_details",
            "finalize": "finalize",
        },
    )

    # After fetching details: finalize
    graph.add_edge("fetch_details", "finalize")

    # Finalize leads to END
    graph.add_edge("finalize", END)

    return graph


def run_pipeline(
    signature: CompanySignature,
    uipath: Optional[UiPath],
    settings: Settings
) -> PipelineState:
    """
    Run the complete PEPPOL search pipeline for a single signature.

    Parameters
    ----------
    signature : CompanySignature
        Company signature to process
    uipath : Optional[UiPath]
        UiPath SDK instance
    settings : Settings
        Configuration settings

    Returns
    -------
    PipelineState
        Final pipeline state with all results
    """
    # Build graph with injected dependencies
    graph = build_pipeline_graph(uipath, settings)
    compiled_graph = graph.compile()

    # Initialize state
    initial_state: PipelineState = {
        "signature": signature,
        "naive_found": False,
        "refined_found": False,
        "peppol_found": False,
    }

    # Run pipeline
    final_state = compiled_graph.invoke(initial_state)

    return final_state
