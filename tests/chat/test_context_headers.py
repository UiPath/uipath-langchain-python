import os
from unittest.mock import patch

from uipath_langchain.chat._headers import build_uipath_context_headers


class TestBuildUiPathContextHeaders:
    """Verify build_uipath_context_headers returns correct headers from UiPathConfig."""

    def test_all_headers_present(self) -> None:
        env = {
            "UIPATH_JOB_KEY": "job-123",
            "UIPATH_FOLDER_KEY": "folder-456",
            "UIPATH_TRACE_ID": "trace-789",
        }
        with patch.dict(os.environ, env, clear=False):
            headers = build_uipath_context_headers()
        assert headers["x-uipath-jobkey"] == "job-123"
        assert headers["x-uipath-folderkey"] == "folder-456"
        assert headers["x-uipath-traceid"] == "trace-789"

    def test_no_headers_when_env_empty(self) -> None:
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            headers = build_uipath_context_headers()
        assert headers == {}

    def test_partial_headers(self) -> None:
        env = {"UIPATH_JOB_KEY": "job-only"}
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_context_headers()
        assert headers == {"x-uipath-jobkey": "job-only"}
        assert "x-uipath-folderkey" not in headers
        assert "x-uipath-traceid" not in headers
