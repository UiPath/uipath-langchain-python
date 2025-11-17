"""Unit tests for feature flags configuration."""

from unittest.mock import Mock, patch

import pytest

from uipath_agents._config.feature_flags import (
    FeatureFlagsConfig,
    get_feature_flags,
)
from uipath_agents._services.flags_service import FeatureFlagsResponse


class TestFeatureFlagsConfig:
    """Tests for FeatureFlagsConfig class."""

    def test_init(self):
        """Test initialization with flags dictionary."""
        flags = {"EnableUnifiedRuntime": True, "CommunityModelNameOverride": "gpt-4"}
        config = FeatureFlagsConfig(flags)
        assert config._flags == flags

    def test_get_existing_flag(self):
        """Test getting an existing flag value."""
        config = FeatureFlagsConfig({"EnableUnifiedRuntime": True})
        assert config.get("EnableUnifiedRuntime") is True

    def test_get_missing_flag_with_default(self):
        """Test getting a missing flag returns default value."""
        config = FeatureFlagsConfig({})
        assert config.get("NonExistent", "default") == "default"

    def test_get_missing_flag_without_default(self):
        """Test getting a missing flag returns None when no default provided."""
        config = FeatureFlagsConfig({})
        assert config.get("NonExistent") is None

    def test_get_various_types(self):
        """Test getting flags with various value types."""
        config = FeatureFlagsConfig(
            {
                "BoolFlag": True,
                "StringFlag": "gpt-4",
                "DictFlag": {"key": "value"},
                "ListFlag": ["item1", "item2"],
                "NullFlag": None,
            }
        )

        assert config.get("BoolFlag") is True
        assert config.get("StringFlag") == "gpt-4"
        assert config.get("DictFlag") == {"key": "value"}
        assert config.get("ListFlag") == ["item1", "item2"]
        assert config.get("NullFlag") is None

    def test_to_dict(self):
        """Test converting config to dictionary."""
        flags = {"EnableUnifiedRuntime": True, "CommunityModelNameOverride": "gpt-4"}
        config = FeatureFlagsConfig(flags)
        result = config.to_dict()

        assert result == flags
        assert result is not flags  # Should be a copy

    def test_to_dict_returns_copy(self):
        """Test that to_dict returns a copy, not the original."""
        flags = {"EnableUnifiedRuntime": True}
        config = FeatureFlagsConfig(flags)
        result = config.to_dict()

        result["NewFlag"] = False
        assert "NewFlag" not in config._flags

    def test_getitem(self):
        """Test dict-like access using []."""
        config = FeatureFlagsConfig({"EnableUnifiedRuntime": True})
        assert config["EnableUnifiedRuntime"] is True

    def test_getitem_missing_raises_keyerror(self):
        """Test that accessing missing flag with [] raises KeyError."""
        config = FeatureFlagsConfig({})
        with pytest.raises(KeyError):
            _ = config["NonExistent"]

    def test_contains(self):
        """Test 'in' operator for checking flag existence."""
        config = FeatureFlagsConfig({"EnableUnifiedRuntime": True})

        assert "EnableUnifiedRuntime" in config
        assert "NonExistent" not in config


class TestGetFeatureFlags:
    """Tests for get_feature_flags function."""

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_get_feature_flags_success(self, mock_uipath_class, mock_service_class):
        """Test successful feature flags retrieval."""
        mock_uipath = Mock()
        mock_uipath._config = Mock()
        mock_uipath._execution_context = Mock()
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_response = FeatureFlagsResponse(
            flags={
                "EnableUnifiedRuntime": True,
                "CommunityModelNameOverride": "gpt-4",
            }
        )
        mock_service.get_feature_flags.return_value = mock_response
        mock_service_class.return_value = mock_service

        result = get_feature_flags(
            ["EnableUnifiedRuntime", "CommunityModelNameOverride"]
        )

        assert isinstance(result, FeatureFlagsConfig)
        assert result.get("EnableUnifiedRuntime") is True
        assert result.get("CommunityModelNameOverride") == "gpt-4"

        mock_service.get_feature_flags.assert_called_once_with(
            ["EnableUnifiedRuntime", "CommunityModelNameOverride"]
        )

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_get_feature_flags_empty_list(self, mock_uipath_class, mock_service_class):
        """Test retrieving feature flags with empty flags list."""
        mock_uipath = Mock()
        mock_uipath._config = Mock()
        mock_uipath._execution_context = Mock()
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_response = FeatureFlagsResponse(flags={})
        mock_service.get_feature_flags.return_value = mock_response
        mock_service_class.return_value = mock_service

        result = get_feature_flags([])

        assert isinstance(result, FeatureFlagsConfig)
        assert result.to_dict() == {}

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_get_feature_flags_api_error(self, mock_uipath_class, mock_service_class):
        """Test that API errors are raised and logged."""
        mock_uipath = Mock()
        mock_uipath._config = Mock()
        mock_uipath._execution_context = Mock()
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_service.get_feature_flags.side_effect = Exception("API Error")
        mock_service_class.return_value = mock_service

        with pytest.raises(Exception, match="API Error"):
            get_feature_flags(["EnableUnifiedRuntime"])

    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_get_feature_flags_uipath_initialization_error(self, mock_uipath_class):
        """Test that UiPath initialization errors are raised."""
        mock_uipath_class.side_effect = Exception("SDK not configured")

        with pytest.raises(Exception, match="SDK not configured"):
            get_feature_flags(["EnableUnifiedRuntime"])

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_get_feature_flags_passes_config_correctly(
        self, mock_uipath_class, mock_service_class
    ):
        """Test that UiPath config and execution context are passed correctly."""
        mock_config = Mock()
        mock_execution_context = Mock()
        mock_uipath = Mock()
        mock_uipath._config = mock_config
        mock_uipath._execution_context = mock_execution_context
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_response = FeatureFlagsResponse(flags={})
        mock_service.get_feature_flags.return_value = mock_response
        mock_service_class.return_value = mock_service

        get_feature_flags(["EnableUnifiedRuntime"])

        mock_service_class.assert_called_once_with(
            config=mock_config,
            execution_context=mock_execution_context,
        )
