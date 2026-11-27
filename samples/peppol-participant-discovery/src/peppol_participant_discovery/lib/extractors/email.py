"""Extract domain from payload and construct billing email address."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

# Blacklist of common domains that are not company-owned
# These are social media, email providers, and other public services
DOMAIN_BLACKLIST = {
    # Social media
    "facebook.com",
    "linkedin.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "pinterest.com",
    "snapchat.com",
    # Link services
    "linktr.ee",
    "linktree.com",
    "bit.ly",
    "tinyurl.com",
    # Email providers
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "aol.com",
    "protonmail.com",
    # File sharing/cloud
    "dropbox.com",
    "google.com",
    "drive.google.com",
    "docs.google.com",
    "apple.com",
    "icloud.com",
    # Mapping/location services
    "maps.google.com",
    "goo.gl",
    "spatial.io",
    # German specific
    "web.de",
    "gmx.de",
    "gmx.net",
    "t-online.de",
    # French specific
    "orange.fr",
    "free.fr",
    "laposte.net",
    "wanadoo.fr",
    # Dutch specific
    "ziggo.nl",
    "xs4all.nl",
    "planet.nl",
    "hetnet.nl",
}


class ExtractedEmail(BaseModel):
    """Email address extracted or constructed from company signature."""

    billing_email: str = Field(description="Constructed billing email (billing@domain)")
    domain: str = Field(description="Extracted domain")
    source_type: str = Field(
        description="How domain was found: 'email', 'url', or 'none'"
    )
    original_value: Optional[str] = Field(
        default=None, description="Original email or URL found in payload"
    )


def extract_domain_from_email(email: str) -> Optional[str]:
    """
    Extract domain from an email address.

    Parameters
    ----------
    email : str
        Email address to extract domain from.

    Returns
    -------
    Optional[str]
        Domain or None if invalid email.
    """
    # Simple email validation and domain extraction
    match = re.match(r"[^@]+@([^@]+)", email)
    if match:
        return match.group(1).strip()
    return None


def extract_domain_from_url(url: str) -> Optional[str]:
    """
    Extract domain from a URL.

    Parameters
    ----------
    url : str
        URL to extract domain from.

    Returns
    -------
    Optional[str]
        Domain or None if invalid URL.
    """
    try:
        # Add scheme if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]

        # Remove 'www.' prefix if present
        if domain.startswith("www."):
            domain = domain[4:]

        return domain if domain else None
    except Exception:
        return None


def extract_email_from_payload(payload: str) -> Optional[ExtractedEmail]:
    """
    Extract domain from payload and construct billing email.

    Searches for:
    1. Email addresses in the payload
    2. URLs in the payload

    Then constructs billing@domain

    Parameters
    ----------
    payload : str
        Company signature payload text.

    Returns
    -------
    Optional[ExtractedEmail]
        Extracted email information, or None if no domain found.
    """

    # Pattern 1: Find email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    email_matches = re.findall(email_pattern, payload)

    if email_matches:
        # Try each email found until we find one not on blacklist
        for email in email_matches:
            domain = extract_domain_from_email(email)

            if domain and domain.lower() not in DOMAIN_BLACKLIST:
                return ExtractedEmail(
                    billing_email=f"billing@{domain}",
                    domain=domain,
                    source_type="email",
                    original_value=email,
                )

    # Pattern 2: Find URLs
    # Match http://, https://, or www. patterns
    url_pattern = r"(?:https?://)?(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:[/\s]|$)"
    url_matches = re.findall(url_pattern, payload)

    if url_matches:
        # Try each URL found until we find one not on blacklist
        for url_domain in url_matches:
            domain = url_domain.strip()

            # Clean up domain (remove trailing punctuation)
            domain = re.sub(r"[^\w.-]$", "", domain)

            if domain and domain.lower() not in DOMAIN_BLACKLIST:
                return ExtractedEmail(
                    billing_email=f"billing@{domain}",
                    domain=domain,
                    source_type="url",
                    original_value=domain,
                )

    # Pattern 3: More aggressive URL extraction
    # Look for domain-like patterns anywhere in text
    lines = payload.split("\n")
    for line in lines:
        # Check if line looks like a URL or domain
        line = line.strip()

        # Match patterns like "example.com", "www.example.com", "http://example.com"
        if (
            "http://" in line.lower()
            or "https://" in line.lower()
            or "www." in line.lower()
        ):
            # Extract just the URL part
            url_match = re.search(
                r"(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}[^\s]*)",
                line,
            )
            if url_match:
                url = url_match.group(0)
                domain = extract_domain_from_url(url)

                if domain and domain.lower() not in DOMAIN_BLACKLIST:
                    return ExtractedEmail(
                        billing_email=f"billing@{domain}",
                        domain=domain,
                        source_type="url",
                        original_value=url,
                    )

    # No domain found
    return None


def validate_domain(domain: str) -> bool:
    """
    Validate that a domain looks reasonable.

    Parameters
    ----------
    domain : str
        Domain to validate.

    Returns
    -------
    bool
        True if domain looks valid.
    """
    # Basic validation
    if not domain or len(domain) < 4:  # e.g., "a.co"
        return False

    # Must have at least one dot
    if "." not in domain:
        return False

    # Must not start or end with dot
    if domain.startswith(".") or domain.endswith("."):
        return False

    # Must have valid characters
    if not re.match(r"^[a-zA-Z0-9.-]+$", domain):
        return False

    return True
