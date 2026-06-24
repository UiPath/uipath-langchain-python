"""State definitions for LangGraph pipeline."""

from __future__ import annotations

from typing import Optional, TypedDict

from ..models import (
    CompanySignature,
    ExtractedData,
    PeppolLookupResult,
    PeppolParticipantDetails,
)


class PipelineState(TypedDict, total=False):
    """
    State for the address normalization pipeline with PEPPOL lookup.

    This state is passed between nodes in the LangGraph StateGraph.
    """

    # Input data
    signature: CompanySignature
    """Original company signature from API."""

    # Extraction stage
    extracted_data: Optional[ExtractedData]
    """Extracted company data (name, country, email, domain)."""

    # PEPPOL search - Attempt 1 (naive)
    peppol_naive_result: Optional[PeppolLookupResult]
    """PEPPOL search result with raw company name."""

    naive_found: bool
    """Whether naive PEPPOL search succeeded."""

    # Refinement stage (only if naive search failed)
    refined_company_name: Optional[str]
    """LLM-refined company name for retry."""

    # PEPPOL search - Attempt 2 (refined)
    peppol_refined_result: Optional[PeppolLookupResult]
    """PEPPOL search result with refined company name."""

    refined_found: bool
    """Whether refined PEPPOL search succeeded."""

    # Final result
    peppol_found: bool
    """Whether PEPPOL participant was found (either attempt)."""

    peppol_participant_details: Optional[PeppolParticipantDetails]
    """Detailed PEPPOL participant data with raw JSON (for reviewer inspection)."""

    # Output/storage
    output_path: Optional[str]
    """Path where results were saved."""

    # Error handling
    error: Optional[str]
    """Error message if pipeline failed."""
