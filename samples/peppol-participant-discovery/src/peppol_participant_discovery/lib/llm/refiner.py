"""LLM-based company name refiner for improved PEPPOL search results."""

from __future__ import annotations

from typing import Optional

from uipath import UiPath

from ..config import Settings
from .prompts import format_refine_prompt
from .uipath_llm import call_llm_with_fallback


def refine_company_name(
    company_name: str,
    country: Optional[str],
    uipath: Optional[UiPath],
    settings: Settings,
) -> str:
    """
    Refine company name for better PEPPOL search results.

    Uses LLM to remove legal suffixes and normalize the name.

    Parameters
    ----------
    company_name : str
        Original company name from payload.
    country : Optional[str]
        Country code (helps with country-specific legal forms).
    uipath : Optional[UiPath]
        UiPath SDK instance (if available).
    settings : Settings
        Configuration settings.

    Returns
    -------
    str
        Refined company name.
    """
    # Format prompt (with bucket support if configured)
    prompt_text = format_refine_prompt(
        company_name=company_name,
        country=country,
        bucket_url=getattr(settings, 'prompt_bucket_url', None),
        api_key=getattr(settings, 'prompt_bucket_api_key', None)
    )

    # Call LLM
    try:
        refined = call_llm_with_fallback(
            uipath=uipath,
            messages=[("system", prompt_text)],
            model=settings.model,
            temperature=0.0,  # Deterministic for refinement
            max_tokens=256,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
        )

        # Clean up any whitespace
        refined = refined.strip()

        # Fallback: if LLM returns empty or very short, use original
        if not refined or len(refined) < 3:
            return company_name

        return refined

    except Exception as e:
        print(f"Company name refinement failed: {e}, using original name")
        return company_name
