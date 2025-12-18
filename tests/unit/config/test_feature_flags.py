"""Unit tests for feature flags."""

from unittest.mock import Mock, patch

import pytest

from uipath_agents._config.feature_flags import _fetch_flags, get_flags
from uipath_agents._services.flags_service import FeatureFlagsResponse


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru_cache before each test."""
    _fetch_flags.cache_clear()
    yield
    _fetch_flags.cache_clear()


class TestGetFlags:
    """Tests for get_flags function."""

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_returns_flags_dict(self, mock_uipath_class, mock_service_class):
        """Test returns raw flags dictionary."""
        mock_uipath = Mock()
        mock_uipath._config = Mock()
        mock_uipath._execution_context = Mock()
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_response = FeatureFlagsResponse(
            flags={"FlagA": True, "FlagB": [1, 2, 3], "FlagC": {"nested": "value"}}
        )
        mock_service.get_feature_flags.return_value = mock_response
        mock_service_class.return_value = mock_service

        result = get_flags(["FlagA", "FlagB", "FlagC"])

        assert result == {
            "FlagA": True,
            "FlagB": [1, 2, 3],
            "FlagC": {"nested": "value"},
        }

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_returns_empty_dict(self, mock_uipath_class, mock_service_class):
        """Test returns empty dict when no flags requested."""
        mock_uipath = Mock()
        mock_uipath._config = Mock()
        mock_uipath._execution_context = Mock()
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_response = FeatureFlagsResponse(flags={})
        mock_service.get_feature_flags.return_value = mock_response
        mock_service_class.return_value = mock_service

        result = get_flags([])

        assert result == {}

    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_propagates_sdk_error(self, mock_uipath_class):
        """Test SDK errors propagate to caller."""
        mock_uipath_class.side_effect = Exception("SDK not configured")

        with pytest.raises(Exception, match="SDK not configured"):
            get_flags(["TestFlag"])

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_caches_results(self, mock_uipath_class, mock_service_class):
        """Test flags are cached after first fetch."""
        mock_uipath = Mock()
        mock_uipath._config = Mock()
        mock_uipath._execution_context = Mock()
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_response = FeatureFlagsResponse(flags={"Flag": "value"})
        mock_service.get_feature_flags.return_value = mock_response
        mock_service_class.return_value = mock_service

        result1 = get_flags(["Flag"])
        result2 = get_flags(["Flag"])

        assert result1 == result2
        mock_service.get_feature_flags.assert_called_once()

    @patch("uipath_agents._config.feature_flags.FlagsService")
    @patch("uipath_agents._config.feature_flags.UiPath")
    def test_normalizes_flag_order(self, mock_uipath_class, mock_service_class):
        """Test same flags in different order use same cache entry."""
        mock_uipath = Mock()
        mock_uipath._config = Mock()
        mock_uipath._execution_context = Mock()
        mock_uipath_class.return_value = mock_uipath

        mock_service = Mock()
        mock_response = FeatureFlagsResponse(flags={"A": 1, "B": 2})
        mock_service.get_feature_flags.return_value = mock_response
        mock_service_class.return_value = mock_service

        get_flags(["B", "A"])
        get_flags(["A", "B"])

        mock_service.get_feature_flags.assert_called_once()
