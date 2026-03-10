"""Tests for licenseRefId generation, propagation, and header injection."""

from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest

from uipath_agents._observability.llmops.spans.spans_schema.base import (
    license_ref_id_context,
)
from uipath_agents._observability.tracing import _httpx_request_hook

# span_exporter and callback fixtures come from conftest.py


@pytest.fixture(autouse=True)
def _clean_license_ref_id_context():
    """Ensure license_ref_id_context is clean before and after each test."""
    license_ref_id_context.set(None)
    yield
    license_ref_id_context.set(None)


class TestLicenseRefIdOnCompletionSpan:
    """Tests that licenseRefId is generated and attached to model_run spans."""

    def test_license_ref_id_set_on_completion_span(self, callback, span_exporter):
        """Model run span should have a licenseRefId attribute after on_llm_end."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        spans = span_exporter.get_finished_spans()
        model_span = next(s for s in spans if s.name == "Model run")

        assert "licenseRefId" in model_span.attributes
        # Should be a valid UUID string
        license_ref_id = model_span.attributes["licenseRefId"]
        assert isinstance(license_ref_id, str)
        assert len(license_ref_id) == 36  # UUID format: 8-4-4-4-12

    def test_license_ref_id_unique_per_llm_call(self, callback, span_exporter):
        """Each LLM call should get a distinct licenseRefId."""
        run_id_1 = uuid4()
        run_id_2 = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt1"], run_id=run_id_1)
        callback.on_llm_end(None, run_id=run_id_1)

        callback.on_llm_start(serialized, ["prompt2"], run_id=run_id_2)
        callback.on_llm_end(None, run_id=run_id_2)

        spans = span_exporter.get_finished_spans()
        model_spans = [s for s in spans if s.name == "Model run"]
        assert len(model_spans) == 2

        ref_id_1 = model_spans[0].attributes["licenseRefId"]
        ref_id_2 = model_spans[1].attributes["licenseRefId"]
        assert ref_id_1 != ref_id_2


class TestLicenseRefIdContextLifecycle:
    """Tests that the license_ref_id ContextVar is properly managed."""

    def test_license_ref_id_context_cleared_after_llm_end(
        self, callback, span_exporter
    ):
        """ContextVar should be None after on_llm_end completes."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_end(None, run_id=run_id)

        assert license_ref_id_context.get() is None

    def test_license_ref_id_context_cleared_after_llm_error(
        self, callback, span_exporter
    ):
        """ContextVar should be None after on_llm_error completes."""
        run_id = uuid4()
        serialized = {"kwargs": {"model_name": "gpt-4"}}

        callback.on_llm_start(serialized, ["prompt"], run_id=run_id)
        callback.on_llm_error(ValueError("LLM error"), run_id=run_id)

        assert license_ref_id_context.get() is None


class TestLicenseRefIdHeaderInjection:
    """Tests for the httpx request hook that injects X-UiPath-License-RefId."""

    def test_license_ref_id_matches_request_header(self):
        """When license_ref_id_context is set, the hook should inject the header."""
        token = license_ref_id_context.set("test-license-ref-id")
        try:
            request = httpx.Request("GET", "https://example.com/api")
            _httpx_request_hook(MagicMock(), request)

            assert request.headers["X-UiPath-License-RefId"] == "test-license-ref-id"
        finally:
            license_ref_id_context.reset(token)

    def test_no_header_injected_outside_llm_call(self):
        """When license_ref_id_context is None, no header should be injected."""
        # Ensure context is clean
        assert license_ref_id_context.get() is None

        request = httpx.Request("GET", "https://example.com/api")
        _httpx_request_hook(MagicMock(), request)

        assert "X-UiPath-License-RefId" not in request.headers
