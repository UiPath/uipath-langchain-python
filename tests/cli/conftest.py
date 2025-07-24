import pytest


@pytest.fixture
def mock_env_vars():
    return {
        "UIPATH_URL": "http://example.com",
        "UIPATH_ACCESS_TOKEN": "***",
    }
