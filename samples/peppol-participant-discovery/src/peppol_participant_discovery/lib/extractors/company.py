"""Extract company name and person information from signature payloads."""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field


class ExtractedCompanyInfo(BaseModel):
    """Company information extracted from signature."""

    company_name: str = Field(description="Company name")


def extract_company_info(payload: str) -> ExtractedCompanyInfo:
    """
    Extract company name from signature payload.

    The payload typically follows these patterns:

    OpenCorporates format:
        Company Name
        Address
        Registration Number

    Coresignal format:
        Company Name
        URL
        Email
        Reg-No: number

    Parameters
    ----------
    payload : str
        Company signature payload text.

    Returns
    -------
    ExtractedCompanyInfo
        Extracted company information.
    """

    lines = [line.strip() for line in payload.split("\n") if line.strip()]

    if not lines:
        return ExtractedCompanyInfo(company_name="Unknown")

    # Registration number patterns (indicates NOT a company name)
    reg_patterns = [
        r"^HRB\s+\d+",
        r"^HRA\s+\d+",
        r"^Reg-No:\s*\d+",
        r"^\d{5,}$",  # Just a number
    ]

    # Role indicators that suggest a line is person, not company
    # We use this to SKIP person lines and get company name from next line
    role_indicators = [
        # German
        "geschäftsführer",
        "geschäftsführerin",
        "prokurist",
        "prokuristin",
        "vorstand",
        "direktor",
        # French
        "directeur",
        "directrice",
        "président",
        "gérant",
        # Dutch
        "bestuurder",
        # English
        "ceo",
        "cfo",
        "director",
        "managing director",
    ]

    company_name = None

    # Check if first line is person (Name, Role format) - if so, use line 2 as company
    if len(lines) > 1 and "," in lines[0]:
        parts = lines[0].split(",", 1)
        if len(parts) > 1:
            potential_role = parts[1].strip().lower()
            # If the part after comma contains a role indicator, this is a person line
            if any(indicator in potential_role for indicator in role_indicators):
                # Company name is on line 2
                company_name = lines[1]

    # Try first line as company name if it's not a registration number, address, URL, or email
    if company_name is None and len(lines) > 0:
        # Check if first line looks like a registration number
        is_reg_number = any(
            re.match(pattern, lines[0], re.IGNORECASE) for pattern in reg_patterns
        )
        # Check if first line looks like an address (has postal code)
        has_postal = bool(re.search(r"\b\d{4,5}\s*[A-Z]{0,2}\b", lines[0]))
        # Skip URLs
        is_url = "http" in lines[0].lower() or "www." in lines[0].lower()
        # Skip emails
        is_email = "@" in lines[0]

        if not is_reg_number and not has_postal and not is_url and not is_email:
            company_name = lines[0]

    # Fallback: if still no company name, use first non-registration line
    if company_name is None:
        for line in lines:
            is_reg = any(re.match(pattern, line, re.IGNORECASE) for pattern in reg_patterns)
            # Skip lines that look like addresses (have postal codes)
            has_postal = bool(re.search(r"\b\d{4,5}\s*[A-Z]{0,2}\b", line))
            # Skip URLs
            is_url = "http" in line.lower() or "www." in line.lower()
            # Skip emails
            is_email = "@" in line

            if not is_reg and not has_postal and not is_url and not is_email:
                company_name = line
                break

    # Final fallback
    if company_name is None:
        company_name = lines[0] if lines else "Unknown"

    # Clean up company name
    company_name = clean_company_name(company_name)

    return ExtractedCompanyInfo(company_name=company_name)


def clean_company_name(name: str) -> str:
    """
    Clean up company name by removing extra markers.

    Parameters
    ----------
    name : str
        Raw company name.

    Returns
    -------
    str
        Cleaned company name.
    """

    # Remove leading/trailing whitespace
    name = name.strip()

    # Remove "c/o" prefix if present
    if name.lower().startswith("c/o "):
        name = name[4:].strip()

    return name
