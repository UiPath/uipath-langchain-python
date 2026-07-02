"""Normalize LLM provider HTTP errors into a common shape.

Providers behind the LLM Gateway each raise a different exception type, but all
carry the same gateway body (``{status, detail, ...}``); only the attribute that
holds it differs. LangChain may wrap the SDK error, so the useful fields can sit
a few links down the ``__cause__`` chain — always together, on one link.
"""

from dataclasses import dataclass


@dataclass
class ProviderError:
    """Normalized provider HTTP error: status code + user-facing detail."""

    status_code: int | None = None
    detail: str | None = None

    def __bool__(self) -> bool:
        """Truthy once we have a status code — the signal that a provider matched."""
        return self.status_code is not None


def _int(value: object) -> int | None:
    """The value if it is an int (a real HTTP status), else None.

    Guards against matching unrelated exceptions that happen to carry a
    ``code``/``status_code`` attribute that isn't an HTTP status.
    """
    return value if isinstance(value, int) else None


def _detail(body: object) -> str | None:
    """The gateway ``detail`` message from a parsed body dict, if present."""
    if isinstance(body, dict):
        return body.get("detail")

    return None


# One extractor per provider: read that SDK's status code and detail out of the exception, if present
# Returns an empty (falsy) ProviderError when ``e`` isn't that provider's error type


def _from_openai(e: BaseException) -> ProviderError:
    """OpenAI / Anthropic: ``e.status_code`` + ``e.body``."""
    return ProviderError(
        _int(getattr(e, "status_code", None)), _detail(getattr(e, "body", None))
    )


def _from_vertex(e: BaseException) -> ProviderError:
    """Vertex / google.genai ``APIError``."""
    return ProviderError(
        _int(getattr(e, "code", None)), _detail(getattr(e, "details", None))
    )


def _from_bedrock(e: BaseException) -> ProviderError:
    """Bedrock — same ``e.status_code`` + ``e.body`` shape as OpenAI.

    Bedrock requests go through the uipath-client ``WrappedBotoClient`` shim
    rather than boto3. On a gateway HTTP error its ``raise_for_status`` raises a
    ``UiPathPermissionDeniedError`` (a ``UiPathAPIError`` / ``httpx.HTTPStatusError``
    subclass) that exposes the OpenAI-style ``.status_code`` and ``.body``.
    """
    return ProviderError(
        _int(getattr(e, "status_code", None)), _detail(getattr(e, "body", None))
    )


def _from_botocore(e: BaseException) -> ProviderError:
    """Bedrock via legacy direct boto3 (``use_new_llm_clients=False``): a
    ``botocore.exceptions.ClientError`` carrying everything in ``e.response``."""
    resp = getattr(e, "response", None)
    if not isinstance(resp, dict):
        return ProviderError()
    return ProviderError(
        _int(resp.get("ResponseMetadata", {}).get("HTTPStatusCode")),
        _detail(resp.get("Error")),
    )


_PROVIDERS = (_from_openai, _from_vertex, _from_bedrock, _from_botocore)


def extract_provider_error(e: BaseException | None) -> ProviderError:
    """Return the first provider that matches ``e`` or any of its ``__cause__`` links."""
    if e is None:
        return ProviderError()
    for extract in _PROVIDERS:
        error = extract(e)
        if error:
            return error
    return extract_provider_error(e.__cause__)
