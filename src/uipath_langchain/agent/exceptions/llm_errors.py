"""Normalize provider LLM exceptions to EnrichedException."""

import json

import httpx
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

# Sentinel path ensuring the LLM Gateway extractor is selected
# (the extractor router checks for "/llm/" in the URL).
_LLM_SENTINEL_PATH = "/llm/"

LLM_KNOWN_ERRORS: dict[tuple[int, str | None], tuple[str, UiPathErrorCategory]] = {
    (403, "10000"): (
        "License not available for LLM usage. Please check your Agent Units allocation.",
        UiPathErrorCategory.DEPLOYMENT,
    ),
    (400, "context_length_exceeded"): (
        "The conversation exceeded the model's context window. "
        "Consider reducing the number of tools, shortening the system prompt, "
        "or using a model with a larger context window.",
        UiPathErrorCategory.USER,
    ),
}


def normalize_to_enriched(e: Exception) -> EnrichedException | None:
    """Normalize a provider LLM exception to an EnrichedException.

    Detects OpenAI, Bedrock, and Vertex exceptions via duck-typing.
    OpenAI and Vertex already carry an httpx.Response — we use it directly.
    Bedrock only exposes a dict, so an httpx.Response is fabricated.

    Also checks ``__cause__`` to handle LangChain wrappers (e.g.
    ``ChatGoogleGenerativeAIError``) that chain the real provider exception.

    Returns None if the exception is not a recognized LLM HTTP error.
    """
    cause: BaseException | None = e
    while isinstance(cause, Exception):
        http_error = _to_http_status_error(cause)
        if http_error is not None:
            return EnrichedException(http_error)
        cause = cause.__cause__
    return None


def _to_http_status_error(e: Exception) -> httpx.HTTPStatusError | None:
    """Convert a provider exception to an httpx.HTTPStatusError.

    Detection order:
    1. OpenAI — has ``status_code`` (int) + ``response`` (httpx.Response)
    2. Bedrock — has ``response`` (dict with ResponseMetadata)
    3. Vertex — has ``code`` (int) + ``status`` (str) + ``message``
    """
    response = getattr(e, "response", None)

    # OpenAI (APIStatusError) — ``status_code`` distinguishes it from Vertex,
    # which uses ``code`` instead.
    if (
        isinstance(response, httpx.Response)
        and hasattr(e, "status_code")
        and isinstance(e.status_code, int)
    ):
        return httpx.HTTPStatusError(
            message=str(e),
            request=response.request,
            response=response,
        )

    # Bedrock (botocore.ClientError) — response is a dict, not an HTTP object.
    # PascalCase keys (Code, Message) are handled by the LLM Gateway
    # extractor's swapcase lookup in get_typed_field.
    if isinstance(response, dict) and "ResponseMetadata" in response:
        metadata = response["ResponseMetadata"]
        status_code = metadata.get("HTTPStatusCode")
        if isinstance(status_code, int):
            error_dict = response.get("Error", {})
            body_text = json.dumps(error_dict) if error_dict else None
            return _fabricate_http_error(status_code, body_text)

    # Vertex (google.genai.errors.APIError) — uses ``code``/``status`` instead
    # of ``status_code``. Carries an httpx.Response when using the httpx
    # transport; falls back to fabrication for aiohttp or test stubs.
    # Requiring ``message`` avoids false-positives on OSError and similar.
    code = getattr(e, "code", None)
    status = getattr(e, "status", None)
    if isinstance(code, int) and isinstance(status, str) and hasattr(e, "message"):
        if isinstance(response, httpx.Response):
            return httpx.HTTPStatusError(
                message=str(e),
                request=response.request,
                response=response,
            )
        message = getattr(e, "message", None) or str(e)
        body_text = json.dumps({"error": {"message": message}})
        return _fabricate_http_error(code, body_text)

    return None


def _fabricate_http_error(status_code: int, body: str | None) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError from raw status code and body text."""
    response = httpx.Response(
        status_code=status_code,
        content=body.encode("utf-8") if body else b"",
        headers={"content-type": "application/json"} if body else {},
        request=httpx.Request("POST", _LLM_SENTINEL_PATH),
    )
    return httpx.HTTPStatusError(
        message=f"HTTP {status_code}",
        request=response.request,
        response=response,
    )
