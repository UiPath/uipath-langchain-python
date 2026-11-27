"""Simple payload extractor for PEPPOL search.

This module extracts basic information needed for PEPPOL Directory queries.
No complex parsing, no country-specific logic - just simple extraction.
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import ExtractedData
from .company import extract_company_info
from .email import extract_email_from_payload


def extract_country_code(payload: str) -> Optional[str]:
    """
    Extract country code from payload text.

    Simple approach: Look for 2-letter country codes in common positions.

    Parameters
    ----------
    payload : str
        Company signature payload.

    Returns
    -------
    Optional[str]
        Country code (DE, FR, NL, AT, BE, etc.) or None if not found.
    """
    # Country-specific extraction code could be here
    # For now, simple text search for common European country codes

    # Common PEPPOL countries
    peppol_countries = [
        "DE",  # Germany
        "FR",  # France
        "NL",  # Netherlands
        "AT",  # Austria
        "BE",  # Belgium
        "ES",  # Spain
        "IT",  # Italy
        "DK",  # Denmark
        "SE",  # Sweden
        "NO",  # Norway
        "FI",  # Finland
        "PL",  # Poland
        "CH",  # Switzerland
        "GB",  # United Kingdom
        "IE",  # Ireland
    ]

    # Look for country codes at end of address lines or standalone
    # Pattern: space + 2 uppercase letters + (space/newline/period/end)
    for country in peppol_countries:
        # Check for country code in various positions
        patterns = [
            rf"\b{country}\s*$",  # End of line
            rf"\b{country}\.",  # Followed by period
            rf"\s{country}\s",  # Surrounded by spaces
            rf"\s{country}\n",  # End of line before newline
        ]

        for pattern in patterns:
            if re.search(pattern, payload, re.MULTILINE | re.IGNORECASE):
                return country

    return None


def extract_payload_data(payload: str) -> ExtractedData:
    """
    Extract basic data from company signature payload.

    Single extraction pass - gets company name, country, email/domain.
    No complex parsing, no LLM calls.

    Parameters
    ----------
    payload : str
        Company signature payload text.

    Returns
    -------
    ExtractedData
        Extracted company information for PEPPOL search.
    """
    # Extract company name (using existing simple extractor)
    company_info = extract_company_info(payload)
    company_name = company_info.company_name

    # Extract country code
    country = extract_country_code(payload)

    # Extract email/domain (using existing extractor)
    email_info = extract_email_from_payload(payload)
    email = email_info.billing_email if email_info else None
    domain = email_info.domain if email_info else None

    return ExtractedData(
        company_name=company_name,
        country=country,
        email=email,
        domain=domain,
        full_payload=payload,
    )
