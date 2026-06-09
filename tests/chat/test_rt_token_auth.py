"""Reference-token (``rt_``) support in ``PlatformSettings``.

UiPath issues two kinds of access tokens: JWT bearer tokens, and opaque
*reference tokens* (prefixed ``rt_``) that carry no client-readable claims.

Before ``uipath-langchain-client`` 1.13.1 the platform settings validator
decoded *every* token as a JWT (splitting on ``.`` and base64-decoding the
payload) to check expiry and pull out the ``client_id``. An opaque ``rt_``
token has no dot-separated payload, so that decode raised ``ValueError`` and
an agent authenticating with a reference token could never construct a chat
model. 1.13.1 parses JWT claims best-effort and falls back gracefully for
opaque tokens. These tests pin that behaviour so the dependency cannot
regress underneath us.
"""

import base64
import json
import time
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from uipath_langchain_client.settings import PlatformSettings

# An opaque UiPath reference token: no ``.``-separated JWT payload to decode.
_RT_TOKEN = "rt_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_BASE_URL = "https://alpha.uipath.com/TestOrg/TestTenant"


def _make_jwt(payload: dict[str, Any]) -> str:
    """Build a JWT-shaped token (``header.payload.signature``) for ``payload``."""

    def _segment(obj: dict[str, Any]) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{_segment({'alg': 'none'})}.{_segment(payload)}.signature"


def _make_settings(token: str) -> PlatformSettings:
    return PlatformSettings(
        base_url=_BASE_URL,
        organization_id="TestOrg",
        tenant_id="TestTenant",
        access_token=SecretStr(token),
    )


class TestReferenceTokenSettings:
    """``PlatformSettings`` must accept opaque ``rt_`` reference tokens."""

    def test_reference_token_does_not_raise(self) -> None:
        """An opaque ``rt_`` token passes validation (the 1.13.1 fix)."""
        settings = _make_settings(_RT_TOKEN)

        assert settings.access_token.get_secret_value() == _RT_TOKEN
        # Opaque tokens carry no JWT claims, so no client_id is extracted.
        assert settings.client_id is None

    def test_reference_token_used_as_bearer(self) -> None:
        """The reference token is sent verbatim as the Bearer credential."""
        settings = _make_settings(_RT_TOKEN)
        auth = settings.build_auth_pipeline()

        request = httpx.Request("POST", f"{_BASE_URL}/llm")
        authenticated = next(auth.auth_flow(request))

        assert authenticated.headers["Authorization"] == f"Bearer {_RT_TOKEN}"


class TestJwtTokenStillSupported:
    """Regression guard: JWT handling must be unchanged by the ``rt_`` fix."""

    def test_jwt_claims_extracted(self) -> None:
        """A JWT still has its ``client_id`` claim extracted."""
        token = _make_jwt({"client_id": "the-client-id", "exp": time.time() + 3600})

        settings = _make_settings(token)

        assert settings.client_id == "the-client-id"

    def test_expired_jwt_rejected(self) -> None:
        """An expired JWT is still rejected during validation."""
        token = _make_jwt({"exp": time.time() - 3600})

        with pytest.raises(ValueError, match="expired"):
            _make_settings(token)
