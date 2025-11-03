"""Prompt templates for LLM operations - loaded from cloud storage bucket."""

from __future__ import annotations

import httpx
from typing import Optional
from functools import lru_cache


# Prompt for refining company names
REFINE_PROMPT_DEFAULT = """You are a company name normalizer for PEPPOL Directory searches.

Your task: Clean and normalize a company name to improve search results.

Rules:
1. Remove legal entity suffixes: GmbH, AG, Ltd, Inc, LLC, SARL, B.V., etc.
2. Keep the core business name
3. Fix obvious typos or spacing issues
4. Remove special characters that might interfere with search
5. Keep it simple - don't over-normalize

Examples:
- "DANMARC Invest-GmbH" → "DANMARC Invest"
- "Kleopatra GmbH" → "Kleopatra"
- "Example Tech Solutions Ltd." → "Example Tech Solutions"
- "Müller & Schmidt KG" → "Müller Schmidt"

Input company name: {company_name}
Country: {country}

Output ONLY the refined company name, nothing else."""


@lru_cache(maxsize=10)
def load_prompt_from_bucket(
    bucket_url: str,
    prompt_name: str,
    api_key: Optional[str] = None,
    timeout: int = 30
) -> str:
    """
    Load prompt template from cloud storage bucket.

    Parameters
    ----------
    bucket_url : str
        Base URL of the cloud storage bucket (e.g., R2, S3, etc.)
    prompt_name : str
        Name of the prompt file to load
    api_key : Optional[str]
        API key for authentication if required
    timeout : int
        Request timeout in seconds

    Returns
    -------
    str
        Prompt template content

    Raises
    ------
    httpx.HTTPError
        If the request fails
    """
    # Construct full URL
    url = f"{bucket_url.rstrip('/')}/{prompt_name}"

    # Prepare headers
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Fetch prompt
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.text


def get_refine_prompt(
    bucket_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Get company name refinement prompt.

    If bucket_url is provided, loads from bucket.
    Otherwise, returns default embedded prompt.

    Parameters
    ----------
    bucket_url : Optional[str]
        Cloud storage bucket URL
    api_key : Optional[str]
        API key for bucket access

    Returns
    -------
    str
        Prompt template
    """
    if bucket_url:
        try:
            return load_prompt_from_bucket(
                bucket_url,
                "company_name_refine.txt",
                api_key
            )
        except Exception:
            # Fallback to default if bucket load fails
            pass

    return REFINE_PROMPT_DEFAULT


def format_refine_prompt(
    company_name: str,
    country: Optional[str] = None,
    bucket_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Format the company name refinement prompt with actual values.

    Parameters
    ----------
    company_name : str
        Company name to refine
    country : Optional[str]
        Country code
    bucket_url : Optional[str]
        Cloud storage bucket URL for loading prompt template
    api_key : Optional[str]
        API key for bucket access

    Returns
    -------
    str
        Formatted prompt
    """
    template = get_refine_prompt(bucket_url, api_key)

    return template.format(
        company_name=company_name,
        country=country or "Unknown"
    )
