"""PEPPOL Directory API client for querying business entities."""

from __future__ import annotations

import time
from typing import Optional

import httpx

from ..models import (
    PeppolDocType,
    PeppolEntity,
    PeppolLookupResult,
    PeppolMatch,
    PeppolParticipantDetails,
)


class PeppolClient:
    """Client for querying the PEPPOL Directory API."""

    BASE_URL = "https://directory.peppol.eu/search/1.0/json"
    RATE_LIMIT_DELAY = 0.5  # 2 queries per second = 0.5s between queries

    def __init__(self, timeout: int = 30):
        """
        Initialize PEPPOL client.

        Parameters
        ----------
        timeout : int
            HTTP request timeout in seconds.
        """
        self.timeout = timeout
        self._last_query_time: float = 0.0

    def _enforce_rate_limit(self):
        """Enforce rate limit of 2 queries per second."""
        current_time = time.time()
        time_since_last_query = current_time - self._last_query_time

        if time_since_last_query < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - time_since_last_query)

        self._last_query_time = time.time()

    def search_by_country_and_name(
        self,
        country: str,
        company_name: str,
        result_page_count: int = 20,
    ) -> PeppolLookupResult:
        """
        Search PEPPOL Directory by country and company name.

        Parameters
        ----------
        country : str
            ISO country code (e.g., 'DE', 'FR', 'NL', 'AT', 'BE').
        company_name : str
            Company name to search for (partial matches supported).
        result_page_count : int
            Number of results to return per page (default: 20).

        Returns
        -------
        PeppolLookupResult
            Search results with found flag and matches.

        Raises
        ------
        httpx.HTTPError
            If the request fails.
        ValueError
            If company_name is too short (≤ 2 characters).
        """
        # Validate inputs
        if len(company_name) <= 2:
            raise ValueError(
                f"Company name must be > 2 characters, got: '{company_name}'"
            )

        # Enforce rate limit
        self._enforce_rate_limit()

        # Build query parameters
        params = {
            "country": country.upper(),
            "name": company_name,
            "rpc": result_page_count,
        }

        # Make request
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        # Parse response
        total_count = data.get("total-result-count", 0)
        matches_data = data.get("matches", [])

        # Normalize entity names in the raw data before Pydantic validation
        for match in matches_data:
            entities = match.get("entities", [])
            for entity in entities:
                name_value = entity.get("name")
                # Normalize name field if it's a list
                if isinstance(name_value, list):
                    if len(name_value) > 0 and isinstance(name_value[0], dict):
                        entity["name"] = name_value[0].get("name", "Unknown")
                    elif len(name_value) > 0:
                        entity["name"] = str(name_value[0])
                    else:
                        entity["name"] = "Unknown"

        # Convert to our models
        matches = [PeppolMatch(**match) for match in matches_data]

        return PeppolLookupResult(
            found=total_count > 0,
            match_count=len(matches),
            matches=matches,
            query_terms=f"country={country}&name={company_name}",
            **{"total-result-count": total_count},
        )

    def search_by_participant_id(
        self, participant_id: str
    ) -> Optional[PeppolMatch]:
        """
        Search PEPPOL Directory by exact participant ID.

        Parameters
        ----------
        participant_id : str
            Full participant ID with scheme (e.g., 'iso6523-actorid-upis::9915:test').

        Returns
        -------
        Optional[PeppolMatch]
            Match if found, None otherwise.

        Raises
        ------
        httpx.HTTPError
            If the request fails.
        """
        # Enforce rate limit
        self._enforce_rate_limit()

        # Build query parameters
        params = {"participant": participant_id}

        # Make request
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        # Parse response
        matches_data = data.get("matches", [])

        if not matches_data:
            return None

        # Normalize entity names
        for match in matches_data:
            entities = match.get("entities", [])
            for entity in entities:
                name_value = entity.get("name")
                if isinstance(name_value, list):
                    if len(name_value) > 0 and isinstance(name_value[0], dict):
                        entity["name"] = name_value[0].get("name", "Unknown")
                    elif len(name_value) > 0:
                        entity["name"] = str(name_value[0])
                    else:
                        entity["name"] = "Unknown"

        return PeppolMatch(**matches_data[0])

    def get_participant_details(
        self, participant_id: str
    ) -> Optional[PeppolParticipantDetails]:
        """
        Get detailed participant information including raw JSON for reviewer inspection.

        This makes a second call to the PEPPOL Directory to fetch complete
        participant details including address data in the JSON response.

        Parameters
        ----------
        participant_id : str
            Full participant ID with scheme (e.g., 'iso6523-actorid-upis::9915:b').

        Returns
        -------
        Optional[PeppolParticipantDetails]
            Detailed participant information with raw JSON, or None if not found.

        Raises
        ------
        httpx.HTTPError
            If the request fails.
        """
        # Enforce rate limit
        self._enforce_rate_limit()

        # Build query parameters
        params = {"participant": participant_id}

        # Make JSON request
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        # Keep raw JSON for reviewer inspection
        raw_json = data.copy()

        # Parse JSON response
        matches_data = data.get("matches", [])

        if not matches_data:
            return None

        match = matches_data[0]

        # Normalize entity names
        entities_data = match.get("entities", [])
        for entity in entities_data:
            name_value = entity.get("name")
            if isinstance(name_value, list):
                if len(name_value) > 0 and isinstance(name_value[0], dict):
                    entity["name"] = name_value[0].get("name", "Unknown")
                elif len(name_value) > 0:
                    entity["name"] = str(name_value[0])
                else:
                    entity["name"] = "Unknown"

        # Parse entities
        entities = [PeppolEntity(**e) for e in entities_data]

        # Parse doc types
        doc_types_data = match.get("docTypes", [])
        doc_types = [PeppolDocType(**dt) for dt in doc_types_data]

        return PeppolParticipantDetails(
            participant_id=participant_id,
            entities=entities,
            doc_types=doc_types,
            raw_json=raw_json,
        )


def lookup_peppol(
    country: str,
    company_name: str,
    timeout: int = 30,
) -> PeppolLookupResult:
    """
    Convenience function to lookup a company in PEPPOL Directory.

    Parameters
    ----------
    country : str
        ISO country code (e.g., 'DE', 'FR', 'NL').
    company_name : str
        Company name to search for.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    PeppolLookupResult
        Search results.
    """
    client = PeppolClient(timeout=timeout)
    return client.search_by_country_and_name(country, company_name)
