import os
from typing import Any

import httpx
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings
from pydantic import Field
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.utils import EndpointManager

from uipath_langchain._utils._request_mixin import UiPathRequestMixin


class UiPathAzureOpenAIEmbeddings(UiPathRequestMixin, AzureOpenAIEmbeddings):
    """Custom Embeddings connector for LangChain integration with UiPath.

    This class modifies the OpenAI client to:
    - Use UiPath endpoints
    - Log request/response durations
    - Apply custom URL preparation and header building
    """

    model_name: str | None = Field(
        default_factory=lambda: os.getenv(
            "UIPATH_MODEL_NAME", "text-embedding-3-large"
        ),
        alias="model",
    )

    def __init__(self, **kwargs):
        default_client_kwargs = get_httpx_client_kwargs()
        client_kwargs = {
            **default_client_kwargs,
            "event_hooks": {
                "request": [self._log_request_duration],
                "response": [self._log_response_duration],
            },
        }
        aclient_kwargs = {
            **default_client_kwargs,
            "event_hooks": {
                "request": [self._alog_request_duration],
                "response": [self._alog_response_duration],
            },
        }
        super().__init__(
            http_client=httpx.Client(**client_kwargs),
            http_async_client=httpx.AsyncClient(**aclient_kwargs),
            **kwargs,
        )
        # Monkey-patch the OpenAI client to use your custom methods
        self.client._client._prepare_url = self._prepare_url
        self.client._client._build_headers = self._build_headers
        self.async_client._client._prepare_url = self._prepare_url
        self.async_client._client._build_headers = self._build_headers

    @property
    def endpoint(self) -> str:
        endpoint = EndpointManager.get_embeddings_endpoint()
        return endpoint.format(
            model=self.model_name, api_version=self.openai_api_version
        )


class UiPathOpenAIEmbeddings(UiPathRequestMixin, OpenAIEmbeddings):
    """Custom Embeddings connector for LangChain integration with UiPath.

    This implementation uses custom _call and _acall methods for full control
    over the API request/response cycle.
    """

    model_name: str | None = Field(
        default_factory=lambda: os.getenv(
            "UIPATH_MODEL_NAME", "text-embedding-3-large"
        ),
        alias="model",
    )

    # Add instance variables for tracking if needed
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._total_tokens = 0
        self._total_requests = 0

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]:
        """Embed a list of documents using UiPath endpoint.

        Args:
            texts: List of texts to embed
            chunk_size: Number of texts to process in each batch
            **kwargs: Additional arguments passed to the API

        Returns:
            List of embeddings for each text
        """
        chunk_size_ = chunk_size or self.chunk_size
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), chunk_size_):
            chunk = texts[i : i + chunk_size_]

            # Build payload matching OpenAI API format
            payload: dict[str, Any] = {
                "input": chunk,
                "model": self.model,
            }

            # Add optional parameters
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions

            # Add model_kwargs and any additional kwargs
            payload.update(self.model_kwargs)
            payload.update(kwargs)

            # Make the API call using custom _call method
            response = self._call(self.url, payload, self.auth_headers)

            # Extract embeddings
            chunk_embeddings = [r["embedding"] for r in response["data"]]
            embeddings.extend(chunk_embeddings)

            # Track usage internally (optional)
            if "usage" in response:
                self._total_tokens += response["usage"].get("total_tokens", 0)
                self._total_requests += 1

        return embeddings

    async def aembed_documents(
        self,
        texts: list[str],
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Async version of embed_documents.

        Args:
            texts: List of texts to embed
            chunk_size: Number of texts to process in each batch
            **kwargs: Additional arguments passed to the API

        Returns:
            List of embeddings for each text
        """
        chunk_size_ = chunk_size or self.chunk_size
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), chunk_size_):
            chunk = texts[i : i + chunk_size_]

            # Build payload matching OpenAI API format
            payload: dict[str, Any] = {
                "input": chunk,
                "model": self.model,
            }

            # Add optional parameters
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions

            # Add model_kwargs and any additional kwargs
            payload.update(self.model_kwargs)
            payload.update(kwargs)

            # Make the async API call using custom _acall method
            response = await self._acall(self.url, payload, self.auth_headers)

            # Extract embeddings
            chunk_embeddings = [r["embedding"] for r in response["data"]]
            embeddings.extend(chunk_embeddings)

            # Track usage internally (optional)
            if "usage" in response:
                self._total_tokens += response["usage"].get("total_tokens", 0)
                self._total_requests += 1

        return embeddings

    @property
    def endpoint(self) -> str:
        """Get the UiPath endpoint for embeddings."""
        endpoint = EndpointManager.get_embeddings_endpoint()
        return endpoint.format(
            model=self.model_name, api_version=self.openai_api_version
        )

    @property
    def url(self) -> str:
        """Get the full URL for API requests."""
        return self.endpoint

    @property
    def auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {}
        if self.openai_api_key:
            headers["Authorization"] = (
                f"Bearer {self.openai_api_key.get_secret_value()}"
            )
        if self.default_headers:
            headers.update(self.default_headers)
        return headers

    def get_usage_stats(self) -> dict[str, int]:
        """Get token usage statistics.

        Returns:
            Dictionary with total_tokens and total_requests
        """
        return {
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
        }
