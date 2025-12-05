"""Service for retrieving feature flags from the UiPath Agents Runtime API."""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field
from uipath._utils import Endpoint, RequestSpec
from uipath.platform.common import BaseService, UiPathApiConfig, UiPathExecutionContext


class FeatureFlagsRequest(BaseModel):
    """Request payload for the feature flags API."""

    flags: List[str] = Field(description="List of feature flag names to retrieve")

    model_config = ConfigDict(populate_by_name=True)


class FeatureFlagsResponse(BaseModel):
    """Response from the feature flags API."""

    flags: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class FlagsService(BaseService):
    """Service for retrieving feature flags from the Agents Runtime API."""

    def __init__(
        self, config: UiPathApiConfig, execution_context: UiPathExecutionContext
    ) -> None:
        """Initialize the flags service."""
        super().__init__(config=config, execution_context=execution_context)

    def get_feature_flags(self, flags: List[str]) -> FeatureFlagsResponse:
        """Retrieve feature flags from the Agents Runtime API."""
        request_payload = FeatureFlagsRequest(flags=flags)
        payload = request_payload.model_dump(by_alias=True)

        spec = RequestSpec(
            method="POST",
            endpoint=Endpoint("/agentsruntime_/api/featureFlags"),
            json=payload,
        )

        response = self.request(
            spec.method,
            url=spec.endpoint,
            json=spec.json,
            headers=spec.headers,
        )

        response_data = response.json()

        return FeatureFlagsResponse(flags=response_data)
