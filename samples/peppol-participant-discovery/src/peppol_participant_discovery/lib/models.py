"""Pydantic models for the mail signature to PEPPOL pipeline."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ========== Company Signature Models ==========

class SignatureReference(BaseModel):
    """Reference metadata for a company signature."""

    source: str = Field(description="Data source (e.g., opencorporates, coresignal)")
    id: str = Field(description="Unique identifier in the source system")
    lineage_id: str = Field(description="Data lineage tracking identifier")
    href: str = Field(description="Relative path to full dataset record")


class CompanySignature(BaseModel):
    """Company signature data from the API."""

    payload: str = Field(
        description="Unstructured text containing company information and address"
    )
    reference: SignatureReference = Field(description="Source metadata")


class SignatureResponse(BaseModel):
    """API response containing company signatures."""

    count: int = Field(description="Number of items returned")
    items: list[CompanySignature] = Field(description="List of company signatures")


# ========== Extracted Data Models ==========

class ExtractedData(BaseModel):
    """Data extracted from company signature payload for PEPPOL search."""

    company_name: str = Field(description="Company name")
    country: Optional[str] = Field(
        default=None, description="Country code (DE, FR, NL, AT, BE, etc.)"
    )
    email: Optional[str] = Field(default=None, description="Email address")
    domain: Optional[str] = Field(default=None, description="Domain from email/URL")
    full_payload: str = Field(description="Original payload text")


# ========== PEPPOL Models ==========

class PeppolParticipantID(BaseModel):
    """PEPPOL participant identifier."""

    scheme: str = Field(description="Identifier scheme (e.g., iso6523-actorid-upis)")
    value: str = Field(description="Participant identifier value")


class PeppolDocType(BaseModel):
    """PEPPOL document type identifier."""

    scheme: str = Field(description="Document type scheme")
    value: str = Field(description="Document type value")


class PeppolEntity(BaseModel):
    """
    Business entity information from PEPPOL (normalized).

    The PEPPOL API returns name as either a string or a list of objects.
    We normalize it to always be a simple string.
    """

    name: str = Field(description="Business entity name (normalized to string)")
    country_code: str = Field(
        alias="countryCode", description="ISO country code (e.g., DE, AT, BE)"
    )
    geo_info: Optional[str] = Field(
        default=None, alias="geoInfo", description="Geographical information"
    )
    websites: list[str] = Field(default_factory=list, description="Website URLs")
    additional_info: Optional[str] = Field(
        default=None, alias="additionalInfo", description="Additional information"
    )
    reg_date: Optional[str] = Field(
        default=None, alias="regDate", description="Registration date (YYYY-MM-DD)"
    )

    @classmethod
    def model_validate(cls, obj):
        """Normalize the name field before validation."""
        if isinstance(obj, dict):
            name_value = obj.get("name")

            # Normalize name field
            if isinstance(name_value, list):
                # Extract first name from list of name objects
                if len(name_value) > 0 and isinstance(name_value[0], dict):
                    obj["name"] = name_value[0].get("name", "Unknown")
                elif len(name_value) > 0:
                    obj["name"] = str(name_value[0])
                else:
                    obj["name"] = "Unknown"
            elif not isinstance(name_value, str):
                obj["name"] = "Unknown"

        return super().model_validate(obj)

    class Config:
        populate_by_name = True


class PeppolMatch(BaseModel):
    """Single PEPPOL directory match result."""

    participant_id: PeppolParticipantID = Field(
        alias="participantID", description="PEPPOL participant identifier"
    )
    doc_types: list[PeppolDocType] = Field(
        default_factory=list,
        alias="docTypes",
        description="Supported document types",
    )
    entities: list[PeppolEntity] = Field(
        default_factory=list, description="Business entities"
    )

    class Config:
        populate_by_name = True


class PeppolLookupResult(BaseModel):
    """Result from PEPPOL Directory lookup."""

    found: bool = Field(description="Whether any matches were found")
    match_count: int = Field(description="Number of matches found")
    matches: list[PeppolMatch] = Field(
        default_factory=list, description="List of matches"
    )
    query_terms: str = Field(description="Query terms used for search")
    total_result_count: int = Field(
        alias="total-result-count", description="Total number of results"
    )

    class Config:
        populate_by_name = True


class PeppolParticipantDetails(BaseModel):
    """Detailed participant information from PEPPOL Directory (for reviewer inspection)."""

    participant_id: str = Field(description="Full participant ID")
    entities: list[PeppolEntity] = Field(
        default_factory=list, description="All business entities for this participant"
    )
    doc_types: list[PeppolDocType] = Field(
        default_factory=list, description="Supported document types"
    )
    raw_json: Optional[dict] = Field(
        default=None, description="Raw JSON response for reviewer inspection"
    )
