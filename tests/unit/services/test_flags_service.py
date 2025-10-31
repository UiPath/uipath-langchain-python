"""Unit tests for FlagsService."""

from uipath_lowcode._services.flags_service import (
    FeatureFlagsRequest,
    FeatureFlagsResponse,
    FlagsService,
)


class TestFeatureFlagsRequest:
    """Tests for FeatureFlagsRequest model."""

    def test_init(self):
        """Test request model initialization."""
        request = FeatureFlagsRequest(
            flags=["EnableUnifiedRuntime", "CommunityModelNameOverride"]
        )
        assert request.flags == ["EnableUnifiedRuntime", "CommunityModelNameOverride"]

    def test_empty_flags(self):
        """Test request with empty flags list."""
        request = FeatureFlagsRequest(flags=[])
        assert request.flags == []

    def test_model_dump(self):
        """Test model serialization."""
        request = FeatureFlagsRequest(flags=["EnableUnifiedRuntime"])
        dumped = request.model_dump(by_alias=True)
        assert dumped == {"flags": ["EnableUnifiedRuntime"]}


class TestFeatureFlagsResponse:
    """Tests for FeatureFlagsResponse model."""

    def test_init_with_flags(self):
        """Test response model initialization with flags."""
        response = FeatureFlagsResponse(
            flags={
                "EnableUnifiedRuntime": True,
                "CommunityModelNameOverride": "gpt-4",
            }
        )
        assert response.flags["EnableUnifiedRuntime"] is True
        assert response.flags["CommunityModelNameOverride"] == "gpt-4"

    def test_init_empty(self):
        """Test response model with empty flags."""
        response = FeatureFlagsResponse()
        assert response.flags == {}

    def test_various_value_types(self):
        """Test response with various value types."""
        response = FeatureFlagsResponse(
            flags={
                "BoolFlag": True,
                "StringFlag": "value",
                "DictFlag": {"nested": "data"},
                "ListFlag": ["item1", "item2"],
                "NullFlag": None,
            }
        )

        assert response.flags["BoolFlag"] is True
        assert response.flags["StringFlag"] == "value"
        assert response.flags["DictFlag"] == {"nested": "data"}
        assert response.flags["ListFlag"] == ["item1", "item2"]
        assert response.flags["NullFlag"] is None


class TestFlagsService:
    """Tests for FlagsService class.

    Note: FlagsService inherits from BaseService which requires complex initialization.
    Direct unit tests are impractical. Integration tests are covered in test_feature_flags.py
    through the get_feature_flags() function which properly initializes the service.
    """

    def test_class_exists(self):
        """Test that FlagsService class is properly defined."""
        assert FlagsService is not None
        assert hasattr(FlagsService, "get_feature_flags")

    def test_get_feature_flags_method_signature(self):
        """Test that get_feature_flags method has correct signature."""
        import inspect

        sig = inspect.signature(FlagsService.get_feature_flags)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "flags" in params
