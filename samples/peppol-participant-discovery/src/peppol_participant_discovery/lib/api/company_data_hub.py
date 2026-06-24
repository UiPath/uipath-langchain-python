"""API client for fetching company signatures from the Company Data Hub.

Direct HTTP client for the Cloudflare Worker API endpoint.
Authentication via API key configured in environment variables.
"""

from __future__ import annotations

import httpx

from ..config import Settings
from ..models import CompanySignature, SignatureResponse


def fetch_signatures(
    settings: Settings,
    size: int = 1,
    timeout: int = 30,
) -> SignatureResponse:
    """
    Fetch company signatures from the Company Data Hub API.

    Parameters
    ----------
    settings:
        Configuration containing API endpoint and authentication key.
    size:
        Number of signatures to fetch (default: 1).
    timeout:
        Request timeout in seconds (default: 30).

    Returns
    -------
    SignatureResponse:
        Parsed response containing company signatures.

    Raises
    ------
    httpx.HTTPError:
        If the API request fails.
    ValueError:
        If COMPANYDATAHUB_API_KEY is not configured.
    """
    if not settings.companydatahub_api_key:
        raise ValueError(
            "COMPANYDATAHUB_API_KEY not configured. "
            "Set it as an environment variable in UiPath Cloud deployment settings."
        )

    headers = {"X-API-Key": settings.companydatahub_api_key}
    params = {"size": size}

    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            settings.api_endpoint,
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

    return SignatureResponse(**data)


def fetch_single_signature(settings: Settings) -> CompanySignature:
    """
    Convenience function to fetch a single company signature.

    Parameters
    ----------
    settings:
        Configuration containing API endpoint and authentication key.

    Returns
    -------
    CompanySignature:
        A single company signature.

    Raises
    ------
    ValueError:
        If no signatures are returned from the API or if API key is not configured.
    httpx.HTTPError:
        If the API request fails.
    """

    response = fetch_signatures(settings, size=1)

    if not response.items:
        raise ValueError("No signatures returned from API")

    return response.items[0]
