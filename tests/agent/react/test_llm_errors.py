"""Tests for normalize_to_enriched across OpenAI, Bedrock, and Vertex providers."""

import json

import httpx
from uipath.platform.errors import EnrichedException

from uipath_langchain.agent.exceptions import normalize_to_enriched


def _openai_error(
    status_code: int,
    body: dict | str | None = None,
    url: str = "https://gateway.uipath.com/llm/v1/chat/completions",
) -> Exception:
    """Build a fake OpenAI APIStatusError with a real httpx.Response."""
    if isinstance(body, dict):
        content = json.dumps(body).encode()
    elif isinstance(body, str):
        content = body.encode()
    else:
        content = b""
    response = httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": "application/json"} if content else {},
        request=httpx.Request("POST", url),
    )
    exc = type(
        "_OpenAIError",
        (Exception,),
        {
            "status_code": status_code,
            "response": response,
            "body": body,
            "code": body.get("error", {}).get("code")
            if isinstance(body, dict)
            else None,
        },
    )()
    return exc


class TestNormalizeOpenAI:
    """OpenAI SDK exceptions (status_code + httpx.Response)."""

    def test_openai_error_with_dict_body(self):
        exc = _openai_error(
            403, {"errorCode": "10000", "message": "License not available"}
        )
        result = normalize_to_enriched(exc)
        assert result is not None
        assert isinstance(result, EnrichedException)
        assert result.status_code == 403

    def test_openai_error_with_nested_error_code(self):
        exc = _openai_error(
            400,
            {
                "error": {
                    "message": "context too long",
                    "code": "context_length_exceeded",
                }
            },
        )
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 400
        info = result.error_info
        assert info is not None
        assert info.error_code == "context_length_exceeded"

    def test_openai_error_preserves_real_url(self):
        exc = _openai_error(
            500,
            {"error": {"message": "internal"}},
            url="https://llm.example.com/v1/chat",
        )
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 500
        assert "llm.example.com" in result.url

    def test_openai_error_with_string_body(self):
        exc = _openai_error(502, "Bad Gateway")
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 502

    def test_openai_error_with_none_body(self):
        exc = _openai_error(500, None)
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 500


class TestNormalizeBedrock:
    """Bedrock (botocore.ClientError) exceptions."""

    def test_bedrock_error(self):
        exc = type(
            "_BedrockError",
            (Exception,),
            {
                "response": {
                    "ResponseMetadata": {"HTTPStatusCode": 429},
                    "Error": {
                        "Code": "ThrottlingException",
                        "Message": "Rate exceeded",
                    },
                }
            },
        )()
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 429

    def test_bedrock_pascal_case_keys_extracted(self):
        """PascalCase keys from botocore are extracted via swapcase lookup."""
        exc = type(
            "_BedrockError",
            (Exception,),
            {
                "response": {
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                    "Error": {
                        "Code": "10000",
                        "Message": "License not available",
                    },
                }
            },
        )()
        result = normalize_to_enriched(exc)
        assert result is not None
        info = result.error_info
        assert info is not None
        assert info.error_code == "10000"
        assert info.message == "License not available"

    def test_bedrock_error_without_error_dict(self):
        exc = type(
            "_BedrockError",
            (Exception,),
            {
                "response": {
                    "ResponseMetadata": {"HTTPStatusCode": 500},
                }
            },
        )()
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 500

    def test_bedrock_non_dict_response_ignored(self):
        """An exception with a non-dict .response is not Bedrock."""
        exc = type("_Other", (Exception,), {"response": "not a dict"})()
        assert normalize_to_enriched(exc) is None


class TestNormalizeVertex:
    """Vertex AI (google.genai.errors.APIError) exceptions."""

    def test_vertex_error_with_httpx_response(self):
        """Uses the real httpx.Response when available."""
        real_response = httpx.Response(
            403,
            json={"error": {"message": "Permission denied", "code": "10000"}},
            request=httpx.Request("POST", "https://llm.uipath.com/llm/v1/chat"),
        )
        exc = type(
            "_VertexError",
            (Exception,),
            {
                "code": 403,
                "status": "PERMISSION_DENIED",
                "message": "Permission denied",
                "response": real_response,
            },
        )()
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 403
        assert "llm.uipath.com" in result.url

    def test_vertex_error_without_httpx_response(self):
        """Falls back to fabrication when response is not httpx."""
        exc = type(
            "_VertexError",
            (Exception,),
            {
                "code": 403,
                "status": "PERMISSION_DENIED",
                "message": "Permission denied on resource",
            },
        )()
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 403

    def test_vertex_error_falls_back_to_str(self):
        exc = type(
            "_VertexError",
            (Exception,),
            {"code": 500, "status": "INTERNAL", "message": None},
        )("Server error")
        result = normalize_to_enriched(exc)
        assert result is not None
        assert result.status_code == 500


class TestNormalizeUnrecognized:
    """Exceptions that don't match any provider pattern."""

    def test_plain_exception_returns_none(self):
        assert normalize_to_enriched(ValueError("bad")) is None

    def test_runtime_error_returns_none(self):
        assert normalize_to_enriched(RuntimeError("oops")) is None

    def test_non_int_status_code_returns_none(self):
        """status_code must be int for OpenAI detection."""
        exc = type(
            "_Odd",
            (Exception,),
            {
                "status_code": "403",
                "response": httpx.Response(403, request=httpx.Request("POST", "")),
            },
        )()
        assert normalize_to_enriched(exc) is None

    def test_vertex_code_not_int_returns_none(self):
        exc = type(
            "_Odd",
            (Exception,),
            {"code": "403", "status": "DENIED", "message": "x"},
        )()
        assert normalize_to_enriched(exc) is None

    def test_int_code_and_status_without_message_returns_none(self):
        """Prevents false-positives on OSError-like exceptions."""
        exc = type(
            "_NotVertex",
            (Exception,),
            {"code": 403, "status": "DENIED"},
        )()
        assert normalize_to_enriched(exc) is None


class TestNormalizeCauseChain:
    """LangChain wrappers that chain the real provider exception via __cause__."""

    def test_langchain_vertex_wrapper(self):
        """ChatGoogleGenerativeAIError chains the real APIError on __cause__."""
        inner = type(
            "_VertexAPIError",
            (Exception,),
            {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Rate limit exceeded",
            },
        )()
        wrapper = Exception("Error calling model 'gemini-2.5-pro'")
        wrapper.__cause__ = inner

        result = normalize_to_enriched(wrapper)
        assert result is not None
        assert result.status_code == 429

    def test_langchain_vertex_wrapper_with_httpx_response(self):
        """Vertex __cause__ with httpx.Response preserves the real response."""
        real_response = httpx.Response(
            429,
            json={"error": {"message": "Rate limit exceeded"}},
            request=httpx.Request("POST", "https://llm.uipath.com/llm/v1/chat"),
        )
        inner = type(
            "_VertexAPIError",
            (Exception,),
            {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "Rate limit exceeded",
                "response": real_response,
            },
        )()
        wrapper = Exception("Error calling model")
        wrapper.__cause__ = inner

        result = normalize_to_enriched(wrapper)
        assert result is not None
        assert result.status_code == 429
        assert "llm.uipath.com" in result.url

    def test_no_cause_still_returns_none(self):
        """Plain exception without __cause__ returns None."""
        assert normalize_to_enriched(ValueError("bad")) is None

    def test_unrecognized_cause_returns_none(self):
        """__cause__ that doesn't match any provider returns None."""
        wrapper = Exception("wrapper")
        wrapper.__cause__ = ValueError("inner")
        assert normalize_to_enriched(wrapper) is None

    def test_deeply_nested_cause(self):
        """Walks the full __cause__ chain, not just one level."""
        inner = type(
            "_VertexAPIError",
            (Exception,),
            {"code": 500, "status": "INTERNAL", "message": "Server error"},
        )()
        mid = Exception("mid-layer wrapper")
        mid.__cause__ = inner
        outer = Exception("outer wrapper")
        outer.__cause__ = mid

        result = normalize_to_enriched(outer)
        assert result is not None
        assert result.status_code == 500
