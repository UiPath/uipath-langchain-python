"""LangGraph node functions for PEPPOL search pipeline."""

from __future__ import annotations

from typing import Optional

from uipath import UiPath

from ..api.peppol_client import PeppolClient, lookup_peppol
from ..config import Settings
from ..extractors.payload import extract_payload_data
from ..llm.refiner import refine_company_name
from .state import PipelineState


def extract_payload_node(state: PipelineState) -> PipelineState:
    """
    Extract basic data from signature payload.

    Single extraction pass - no complex parsing, no LLM calls.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state with signature.

    Returns
    -------
    PipelineState
        Updated state with extracted_data.
    """
    signature = state["signature"]

    try:
        extracted_data = extract_payload_data(signature.payload)
        state["extracted_data"] = extracted_data
    except Exception as e:
        state["error"] = f"Extraction failed: {str(e)}"

    return state


def peppol_search_naive_node(state: PipelineState) -> PipelineState:
    """
    PEPPOL search attempt #1: Use raw company name.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state with extracted_data.

    Returns
    -------
    PipelineState
        Updated state with peppol_naive_result and naive_found flag.
    """
    extracted = state.get("extracted_data")

    if not extracted or not extracted.company_name or not extracted.country:
        state["naive_found"] = False
        state["error"] = "Missing company name or country for PEPPOL search"
        return state

    try:
        result = lookup_peppol(extracted.country, extracted.company_name)
        state["peppol_naive_result"] = result
        state["naive_found"] = result.found
    except Exception as e:
        state["naive_found"] = False
        state["error"] = f"Naive PEPPOL search failed: {str(e)}"

    return state


def create_refine_company_name_node(
    uipath: Optional[UiPath],
    settings: Settings
):
    """
    Factory function to create refine_company_name_node with injected dependencies.

    Parameters
    ----------
    uipath : Optional[UiPath]
        UiPath SDK instance
    settings : Settings
        Configuration settings

    Returns
    -------
    callable
        Node function with dependencies injected
    """
    def refine_company_name_node(state: PipelineState) -> PipelineState:
        """
        Refine company name using LLM for second PEPPOL search attempt.

        Only called if naive search failed.

        Parameters
        ----------
        state : PipelineState
            Current pipeline state with extracted_data.

        Returns
        -------
        PipelineState
            Updated state with refined_company_name.
        """
        extracted = state.get("extracted_data")

        if not extracted:
            state["refined_company_name"] = None
            return state

        try:
            refined = refine_company_name(
                company_name=extracted.company_name,
                country=extracted.country,
                uipath=uipath,
                settings=settings,
            )
            state["refined_company_name"] = refined
        except Exception as e:
            state["refined_company_name"] = None
            state["error"] = f"Name refinement failed: {str(e)}"

        return state

    return refine_company_name_node


def peppol_search_refined_node(state: PipelineState) -> PipelineState:
    """
    PEPPOL search attempt #2: Use LLM-refined company name.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state with refined_company_name.

    Returns
    -------
    PipelineState
        Updated state with peppol_refined_result and refined_found flag.
    """
    extracted = state.get("extracted_data")
    refined_name = state.get("refined_company_name")

    if not extracted or not refined_name:
        state["refined_found"] = False
        return state

    try:
        result = lookup_peppol(extracted.country, refined_name)
        state["peppol_refined_result"] = result
        state["refined_found"] = result.found
    except Exception as e:
        state["refined_found"] = False
        error_msg = state.get("error", "")
        state["error"] = f"{error_msg}; Refined PEPPOL search failed: {str(e)}"

    return state


def fetch_participant_details_node(state: PipelineState) -> PipelineState:
    """
    Fetch detailed PEPPOL participant information.

    Only called when a PEPPOL participant was found (either naive or refined search).

    Parameters
    ----------
    state : PipelineState
        Current pipeline state with peppol search result.

    Returns
    -------
    PipelineState
        Updated state with peppol_participant_details.
    """
    # Get the successful result (either naive or refined)
    peppol_result = None
    if state.get("naive_found"):
        peppol_result = state.get("peppol_naive_result")
    elif state.get("refined_found"):
        peppol_result = state.get("peppol_refined_result")

    if not peppol_result or not peppol_result.matches:
        state["peppol_participant_details"] = None
        state["peppol_found"] = False
        return state

    # Get first match participant ID
    first_match = peppol_result.matches[0]
    participant_id_full = (
        f"{first_match.participant_id.scheme}::{first_match.participant_id.value}"
    )

    try:
        client = PeppolClient(timeout=30)
        details = client.get_participant_details(participant_id_full)
        state["peppol_participant_details"] = details
        state["peppol_found"] = True
    except Exception as e:
        state["peppol_participant_details"] = None
        state["peppol_found"] = False
        error_msg = state.get("error", "")
        state["error"] = f"{error_msg}; Participant details fetch failed: {str(e)}"

    return state


def finalize_results_node(state: PipelineState) -> PipelineState:
    """
    Finalize results (replaces save_with_peppol and save_without_peppol).

    For UiPath agents, we don't save to files - we just ensure all
    state is properly set for the Output model.

    Parameters
    ----------
    state : PipelineState
        Current pipeline state.

    Returns
    -------
    PipelineState
        State with finalized flags.
    """
    # Just ensure peppol_found is set correctly
    if state.get("naive_found") or state.get("refined_found"):
        state["peppol_found"] = True
    else:
        state["peppol_found"] = False

    return state
