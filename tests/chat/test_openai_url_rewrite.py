import httpx

from uipath_langchain.chat.openai import _rewrite_openai_url


class TestRewriteOpenAIUrl:
    """Tests for the _rewrite_openai_url function."""

    def test_rewrite_deployments_url(self):
        """Test rewriting URLs with /openai/deployments/ pattern (responses: false)."""
        original_url = "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/openai/deployments/gpt-5-mini-2025-08-07/chat/completions?api-version=2024-12-01-preview"
        params = httpx.QueryParams({"api-version": "2024-12-01-preview"})

        result = _rewrite_openai_url(original_url, params)

        assert result is not None
        assert (
            str(result)
            == "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/completions?api-version=2024-12-01-preview"
        )

    def test_rewrite_responses_url(self):
        """Test rewriting URLs with /openai/responses pattern (responses: true)."""
        original_url = "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/openai/responses?api-version=2024-12-01-preview"
        params = httpx.QueryParams({"api-version": "2024-12-01-preview"})

        result = _rewrite_openai_url(original_url, params)

        assert result is not None
        assert (
            str(result)
            == "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/completions?api-version=2024-12-01-preview"
        )

    def test_rewrite_base_url_with_query_params(self):
        """Test rewriting base URL with query params (responses API base case)."""
        original_url = "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07?api-version=2024-12-01-preview"
        params = httpx.QueryParams({"api-version": "2024-12-01-preview"})

        result = _rewrite_openai_url(original_url, params)

        assert result is not None
        assert (
            str(result)
            == "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/completions?api-version=2024-12-01-preview"
        )

    def test_rewrite_without_query_params(self):
        """Test rewriting URL without query parameters."""
        original_url = "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/openai/responses"
        params = httpx.QueryParams()

        result = _rewrite_openai_url(original_url, params)

        assert result is not None
        assert (
            str(result)
            == "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/completions"
        )

    def test_rewrite_localhost_url(self):
        """Test rewriting localhost URL."""
        original_url = "https://localhost:7024/account/tenant/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/openai/deployments/gpt-5-mini-2025-08-07/chat/completions"
        params = httpx.QueryParams()

        result = _rewrite_openai_url(original_url, params)

        assert result is not None
        assert (
            str(result)
            == "https://localhost:7024/account/tenant/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/completions"
        )

    def test_rewrite_preserves_different_api_versions(self):
        """Test that different api-version values are preserved."""
        original_url = "https://cloud.uipath.com/account/tenant/agenthub_/llm/raw/vendor/openai/model/gpt-5-mini-2025-08-07/openai/responses?api-version=2025-04-01-preview"
        params = httpx.QueryParams({"api-version": "2025-04-01-preview"})

        result = _rewrite_openai_url(original_url, params)

        assert result is not None
        assert "api-version=2025-04-01-preview" in str(result)
