from unittest.mock import patch

import pytest
from uipath.core.guardrails import GuardrailValidationResult


@pytest.fixture
def mock_env_vars():
    return {
        "UIPATH_URL": "http://example.com",
        "UIPATH_ACCESS_TOKEN": "***",
        "UIPATH_TENANT_ID": "test-tenant-id",
    }


@pytest.fixture
def mock_guardrails_service():
    """Mock the guardrails service to avoid HTTP errors in tests."""

    def mock_evaluate_guardrail(text, guardrail):
        """Mock guardrail evaluation - always passes validation."""
        return GuardrailValidationResult(validation_passed=True, reason="")

    with patch(
        "uipath.platform.guardrails.GuardrailsService.evaluate_guardrail",
        side_effect=mock_evaluate_guardrail,
    ) as mock:
        yield mock
