"""HTTP client for the Jev (TypeSafe AI) System One API.

Two transports are supported:

- **Direct** (default): calls ``https://api.typesafe.ai/v1/systemone`` with the
  API key from ``TYPESAFE_API_KEY``. Meant for testing until LLM Gateway serves
  Jev.
- **LLM Gateway**: enabled by the ``EnableJevViaLlmGateway`` feature flag. Uses
  the UiPath LLM client settings (AgentHub, Orchestrator or LLM Gateway) and the
  vendor passthrough route
  ``.../raw/vendor/typesafe/model/{model}/systemone``.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.core.feature_flags import FeatureFlags
from uipath.llm_client.settings import (
    ApiType,
    RoutingMode,
    UiPathAPIConfig,
    UiPathBaseSettings,
    get_default_client_settings,
)
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

JEV_VIA_LLM_GATEWAY_FF = "EnableJevViaLlmGateway"

TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
TYPESAFE_BASE_URL_ENV = "TYPESAFE_BASE_URL"
TYPESAFE_DEFAULT_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_SYSTEM_ONE_PATH = "/v1/systemone"

JEV_GATEWAY_VENDOR = "typesafe"
JEV_GATEWAY_API_TYPE = "systemone"

JEV_DEFAULT_TIMEOUT_SECONDS = 30.0

# 429 (rate limited) and 529 (overloaded) are documented as retryable.
_RETRYABLE_STATUS_CODES = frozenset({429, 529})
_MAX_ATTEMPTS = 3
_MAX_RETRY_DELAY_SECONDS = 10.0


@dataclass(frozen=True)
class JevClientConfig:
    """Transport configuration for :class:`JevClient`."""

    use_llm_gateway: bool = False
    api_key: str | None = None
    base_url: str = TYPESAFE_DEFAULT_BASE_URL
    timeout: float = JEV_DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_environment(cls) -> "JevClientConfig":
        """Resolve the configuration from feature flags and environment variables."""
        return cls(
            use_llm_gateway=FeatureFlags.is_flag_enabled(
                JEV_VIA_LLM_GATEWAY_FF, default=False
            ),
            api_key=os.getenv(TYPESAFE_API_KEY_ENV) or None,
            base_url=(
                os.getenv(TYPESAFE_BASE_URL_ENV) or TYPESAFE_DEFAULT_BASE_URL
            ).rstrip("/"),
        )


@dataclass(frozen=True)
class _JevRequestTarget:
    url: str
    headers: dict[str, str]
    auth: httpx.Auth | None


def build_gateway_url(settings: UiPathBaseSettings, model: str) -> str:
    """Build the vendor passthrough URL for Jev on the configured UiPath backend.

    The backends only template completion routes, so the completion passthrough
    URL is built and its API-type suffix swapped for the Jev one. This keeps the
    AgentHub/Orchestrator/LLM Gateway selection and service-URL overrides in one
    place (the LLM client settings).
    """
    url = settings.build_base_url(
        model_name=model,
        api_config=_gateway_api_config(),
    )
    completions_suffix = f"/{ApiType.COMPLETIONS.value}"
    base, _, query = url.partition("?")
    if base.endswith(completions_suffix):
        base = base[: -len(completions_suffix)] + f"/{JEV_GATEWAY_API_TYPE}"
    return f"{base}?{query}" if query else base


def _gateway_api_config() -> UiPathAPIConfig:
    return UiPathAPIConfig(
        api_type=ApiType.COMPLETIONS,
        routing_mode=RoutingMode.PASSTHROUGH,
        vendor_type=JEV_GATEWAY_VENDOR,
        freeze_base_url=True,
    )


class JevClient:
    """Async client for ``POST /v1/systemone``."""

    def __init__(
        self,
        config: JevClientConfig,
        gateway_settings: UiPathBaseSettings | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config: Transport configuration.
            gateway_settings: UiPath LLM client settings used when routing via
                LLM Gateway. Defaults to the environment's settings.
        """
        self._config = config
        self._gateway_settings = gateway_settings

    async def system_one(
        self,
        *,
        state: Any,
        model: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate ``questions`` against ``state`` and return the raw response.

        Raises:
            AgentRuntimeError: On missing configuration, transport errors,
                non-2xx responses, or a malformed response body.
        """
        client_kwargs = get_httpx_client_kwargs()
        platform_headers = client_kwargs.pop("headers", None) or {}
        client_kwargs["timeout"] = self._config.timeout
        target = self._build_target(model, platform_headers)
        payload = {"state": state, "model": model, "questions": questions}

        try:
            async with httpx.AsyncClient(auth=target.auth, **client_kwargs) as client:
                response = await self._post_with_retry(
                    client, target.url, target.headers, payload
                )
        except httpx.TimeoutException as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.HTTP_ERROR,
                title="Jev request timed out",
                detail=f"Jev did not respond within {self._config.timeout}s: {e}",
                category=UiPathErrorCategory.SYSTEM,
            ) from e
        except httpx.HTTPError as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.HTTP_ERROR,
                title="Jev request failed",
                detail=f"Request to Jev failed: {e}",
                category=UiPathErrorCategory.SYSTEM,
            ) from e

        _raise_for_status(response)
        try:
            body = response.json()
        except ValueError as e:
            raise _invalid_response("Response body is not valid JSON.") from e
        if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
            raise _invalid_response("Response has no 'answers' object.")
        return body

    def _build_target(
        self, model: str, platform_headers: dict[str, str]
    ) -> _JevRequestTarget:
        headers = {"Content-Type": "application/json"}

        if self._config.use_llm_gateway:
            settings = self._gateway_settings or get_default_client_settings()
            headers = {
                **platform_headers,
                **settings.build_auth_headers(
                    model_name=model, api_config=_gateway_api_config()
                ),
                **headers,
            }
            return _JevRequestTarget(
                url=build_gateway_url(settings, model),
                headers=headers,
                auth=settings.build_auth_pipeline(),
            )

        if not self._config.api_key:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.HTTP_ERROR,
                title="Jev API key is not configured",
                detail=f"Set the {TYPESAFE_API_KEY_ENV} environment variable to "
                "call Jev directly.",
                category=UiPathErrorCategory.USER,
            )
        # UiPath platform headers (licensing context) are deliberately not sent
        # to the third-party host.
        headers["Authorization"] = f"Bearer {self._config.api_key}"
        return _JevRequestTarget(
            url=f"{self._config.base_url}{TYPESAFE_SYSTEM_ONE_PATH}",
            headers=headers,
            auth=None,
        )

    @staticmethod
    async def _post_with_retry(
        client: httpx.AsyncClient,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            response = await client.post(url, headers=headers, json=payload)
            if (
                response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt == _MAX_ATTEMPTS
            ):
                return response
            await asyncio.sleep(_retry_delay(response, attempt))
        raise AssertionError("unreachable")


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    try:
        delay = float(retry_after) if retry_after is not None else 2.0**attempt
    except ValueError:
        delay = 2.0**attempt
    return max(0.0, min(delay, _MAX_RETRY_DELAY_SECONDS))


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    status = response.status_code
    detail = f"Jev returned HTTP {status}: {response.text[:1000]}"
    if status in (401, 403):
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.HTTP_ERROR,
            title="Jev authentication failed",
            detail=detail,
            category=UiPathErrorCategory.USER,
        )
    if status in (400, 422):
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.HTTP_ERROR,
            title="Jev rejected the request",
            detail=detail,
            category=UiPathErrorCategory.USER,
        )
    raise AgentRuntimeError(
        code=AgentRuntimeErrorCode.HTTP_ERROR,
        title="Jev request failed",
        detail=detail,
        category=UiPathErrorCategory.SYSTEM,
    )


def _invalid_response(detail: str) -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
        title="Invalid response from Jev",
        detail=detail,
        category=UiPathErrorCategory.SYSTEM,
    )
