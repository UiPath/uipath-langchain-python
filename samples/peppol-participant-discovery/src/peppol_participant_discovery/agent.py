"""
Mail Signature to PEPPOL Agent

Extracts company information from email signatures and performs PEPPOL Directory lookup
with a 2-attempt strategy (naive + LLM-refined).
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from uipath import UiPath

from peppol_participant_discovery.lib.api.company_data_hub import fetch_single_signature
from peppol_participant_discovery.lib.config import Settings, load_settings_with_uipath
from peppol_participant_discovery.lib.models import CompanySignature, SignatureReference
from peppol_participant_discovery.lib.pipeline.graph import run_pipeline


# ========== Input Model ==========

class Input(BaseModel):
    """Input for mail signature to PEPPOL conversion"""

    signature_text: Optional[str] = Field(
        default=None,
        description="Email signature text to extract company information from",
        examples=[
            "John Doe, CEO\nAcme Corp\n123 Main St, Berlin 10115 DE\nHRB 12345"
        ]
    )

    fetch_from_api: bool = Field(
        default=False,
        description="Fetch a random signature from Company Data Hub API instead of using signature_text"
    )

    perform_peppol_lookup: bool = Field(
        default=True,
        description="Whether to perform PEPPOL Directory lookup"
    )


# ========== Output Model ==========

class Output(BaseModel):
    """Extracted PEPPOL master data from email signature"""

    # Signature identification
    signature_id: str = Field(
        description="Signature identifier",
        examples=["F1103R_HRB118578", "manual_2025-01-03T10:30:00"]
    )

    source: str = Field(
        description="Data source",
        examples=["opencorporates", "coresignal", "user_input"]
    )

    # Extracted basic data
    company_name: Optional[str] = Field(
        None,
        description="Extracted company name",
        examples=["Bounce GmbH"]
    )

    country: Optional[str] = Field(
        None,
        description="Country code",
        examples=["DE", "FR", "NL"]
    )

    email: Optional[str] = Field(
        None,
        description="Extracted or generated email address",
        examples=["billing@bounce.de"]
    )

    domain: Optional[str] = Field(
        None,
        description="Extracted domain",
        examples=["bounce.de"]
    )

    # PEPPOL lookup results
    peppol_found: bool = Field(
        description="Whether PEPPOL participant was found"
    )

    peppol_participant_id: Optional[str] = Field(
        None,
        description="PEPPOL participant identifier",
        examples=["iso6523-actorid-upis::9915:b"]
    )

    peppol_entities: Optional[List[dict]] = Field(
        None,
        description="PEPPOL business entities"
    )

    # Search metadata
    search_method: Optional[str] = Field(
        None,
        description="Which search method succeeded",
        examples=["naive", "refined"]
    )

    refined_company_name: Optional[str] = Field(
        None,
        description="LLM-refined company name (if refinement was performed)",
        examples=["Bounce"]
    )

    # Status
    validation_status: str = Field(
        description="Overall validation status",
        examples=["valid", "not_found", "error"]
    )

    confidence_score: float = Field(
        description="Confidence score of extraction (0.0-1.0)",
        examples=[1.0, 0.7, 0.0]
    )

    error: Optional[str] = Field(
        None,
        description="Error message if pipeline failed"
    )


# ========== Main Agent Function ==========

def main(input_data: Input) -> Output:
    """
    Main agent function that extracts PEPPOL master data from email signatures.

    Orchestrates a sophisticated 2-attempt PEPPOL lookup pipeline:
    1. Extract company data from signature
    2. PEPPOL search with raw company name (naive)
    3. If not found: LLM-refine company name + retry PEPPOL search
    4. Fetch detailed participant data if found

    Args:
        input_data: Input containing signature text or API fetch flag

    Returns:
        Output with extracted PEPPOL master data
    """

    # Initialize UiPath SDK and settings
    uipath = UiPath()
    settings = load_settings_with_uipath(uipath)

    # ========== Prepare Signature ==========

    signature: CompanySignature

    if input_data.fetch_from_api:
        # Fetch from Company Data Hub API
        try:
            signature = fetch_single_signature(settings)
        except Exception as e:
            # Return error output
            return Output(
                signature_id="error",
                source="api",
                peppol_found=False,
                validation_status="error",
                confidence_score=0.0,
                error=f"Failed to fetch signature from API: {str(e)}"
            )
    else:
        # Use provided signature text
        if not input_data.signature_text:
            return Output(
                signature_id="error",
                source="user_input",
                peppol_found=False,
                validation_status="error",
                confidence_score=0.0,
                error="No signature_text provided and fetch_from_api=False"
            )

        signature = CompanySignature(
            payload=input_data.signature_text,
            reference=SignatureReference(
                source="user_input",
                id=f"manual_{datetime.now().isoformat()}",
                lineage_id="N/A",
                href="N/A"
            )
        )

    # ========== Run Pipeline ==========

    if input_data.perform_peppol_lookup:
        try:
            # Run full LangGraph pipeline
            final_state = run_pipeline(
                signature=signature,
                uipath=uipath,
                settings=settings
            )

            # Extract results from pipeline state
            extracted = final_state.get("extracted_data")
            peppol_details = final_state.get("peppol_participant_details")

            # Determine search method
            search_method = None
            if final_state.get("naive_found"):
                search_method = "naive"
            elif final_state.get("refined_found"):
                search_method = "refined"

            # If PEPPOL participant found, write to queue
            peppol_found = final_state.get("peppol_found", False)
            if peppol_found and peppol_details:
                try:
                    import json
                    import re
                    from uipath.models.queues import CommitType

                    # Create slugified reference from company name and country
                    company_name = extracted.company_name if extracted else ""
                    country = extracted.country if extracted else ""

                    # Slugify: lowercase, replace spaces/special chars with hyphens
                    slug_parts = []
                    if company_name:
                        slug_parts.append(re.sub(r'[^a-z0-9]+', '-', company_name.lower()).strip('-'))
                    if country:
                        slug_parts.append(country.lower())
                    reference = '-'.join(slug_parts) if slug_parts else signature.reference.id

                    # Serialize complex data to JSON strings (queues only support simple types)
                    # Note: When running in UiPath Cloud, folder context is automatic
                    queue_item = {
                        "Name": "mailsig-to-peppol",
                        "Reference": reference,
                        "SpecificContent": {
                            "signature_id": signature.reference.id,
                            "source": signature.reference.source,
                            "signature_payload": signature.payload,
                            "company_name": company_name,
                            "country": country,
                            "email": extracted.email if extracted else "",
                            "domain": extracted.domain if extracted else "",
                            "peppol_participant_id": peppol_details.participant_id,
                            "peppol_entities_json": json.dumps([e.model_dump() for e in peppol_details.entities]),
                            "search_method": search_method or "",
                            "refined_company_name": final_state.get("refined_company_name") or "",
                        }
                    }

                    # Use bulk create_items method
                    uipath.queues.create_items(
                        items=[queue_item],
                        queue_name="mailsig-to-peppol",
                        commit_type=CommitType.ALL_OR_NOTHING
                    )
                except Exception as queue_error:
                    # Don't fail the whole agent, just log the error
                    print(f"Warning: Failed to write to queue: {queue_error}")

            # Build output
            return Output(
                signature_id=signature.reference.id,
                source=signature.reference.source,
                company_name=extracted.company_name if extracted else None,
                country=extracted.country if extracted else None,
                email=extracted.email if extracted else None,
                domain=extracted.domain if extracted else None,
                peppol_found=final_state.get("peppol_found", False),
                peppol_participant_id=(
                    peppol_details.participant_id
                    if peppol_details else None
                ),
                peppol_entities=(
                    [e.model_dump() for e in peppol_details.entities]
                    if peppol_details else None
                ),
                search_method=search_method,
                refined_company_name=final_state.get("refined_company_name"),
                validation_status=(
                    "valid" if final_state.get("peppol_found")
                    else "not_found"
                ),
                confidence_score=(
                    1.0 if final_state.get("naive_found")
                    else 0.7 if final_state.get("refined_found")
                    else 0.0
                ),
                error=final_state.get("error")
            )

        except Exception as e:
            return Output(
                signature_id=signature.reference.id,
                source=signature.reference.source,
                peppol_found=False,
                validation_status="error",
                confidence_score=0.0,
                error=f"Pipeline execution failed: {str(e)}"
            )

    else:
        # Skip PEPPOL lookup - just extract basic data
        from lib.extractors.payload import extract_payload_data

        try:
            extracted = extract_payload_data(signature.payload)

            return Output(
                signature_id=signature.reference.id,
                source=signature.reference.source,
                company_name=extracted.company_name,
                country=extracted.country,
                email=extracted.email,
                domain=extracted.domain,
                peppol_found=False,
                peppol_participant_id=None,
                peppol_entities=None,
                search_method=None,
                refined_company_name=None,
                validation_status="pending",
                confidence_score=0.5,  # Partial - no PEPPOL validation
                error=None
            )

        except Exception as e:
            return Output(
                signature_id=signature.reference.id,
                source=signature.reference.source,
                peppol_found=False,
                validation_status="error",
                confidence_score=0.0,
                error=f"Extraction failed: {str(e)}"
            )
