"""Tests for process tool span attributes."""

from uipath_agents._observability.llmops.spans.span_attributes.tools import (
    ProcessToolSpanAttributes,
)


class TestProcessToolSpanAttributes:
    def test_no_tool_name_field(self) -> None:
        """ProcessToolSpanAttributes should not have toolName in output."""
        attrs = ProcessToolSpanAttributes(
            arguments={"log": "test"},
            job_id="af2c5f8e-3d41-4d17-86e9-0a2e15742ff6",
        )
        otel = attrs.to_otel_attributes()
        assert "toolName" not in otel

    def test_has_job_id_and_job_details_uri(self) -> None:
        """ProcessToolSpanAttributes should have jobId and jobDetailsUri."""
        attrs = ProcessToolSpanAttributes(
            arguments={"log": "test"},
            job_id="af2c5f8e-3d41-4d17-86e9-0a2e15742ff6",
            job_details_uri="https://alpha.uipath.com/.../jobs/af2c5f8e-3d41-4d17-86e9-0a2e15742ff6/details",
        )
        otel = attrs.to_otel_attributes()
        assert otel["jobId"] == "af2c5f8e-3d41-4d17-86e9-0a2e15742ff6"
        assert "jobDetailsUri" in otel
