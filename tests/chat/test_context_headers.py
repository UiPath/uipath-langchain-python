import os
from unittest.mock import patch

from uipath_langchain.chat.http_client import build_uipath_headers


class TestBuildUiPathHeaders:
    """Verify build_uipath_headers returns correct headers from env vars."""

    def test_all_context_headers_present(self) -> None:
        env = {
            "UIPATH_JOB_KEY": "job-123",
            "UIPATH_FOLDER_KEY": "folder-456",
            "UIPATH_TRACE_ID": "trace-789",
        }
        with patch.dict(os.environ, env, clear=False):
            headers = build_uipath_headers("fake-token")
        assert headers["Authorization"] == "Bearer fake-token"
        assert headers["x-uipath-jobkey"] == "job-123"
        assert headers["x-uipath-folderkey"] == "folder-456"
        assert headers["x-uipath-traceid"] == "trace-789"

    def test_no_context_headers_when_env_empty(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            headers = build_uipath_headers("fake-token")
        assert headers == {"Authorization": "Bearer fake-token"}

    def test_partial_context_headers(self) -> None:
        env = {"UIPATH_JOB_KEY": "job-only"}
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_headers("fake-token")
        assert headers["x-uipath-jobkey"] == "job-only"
        assert "x-uipath-folderkey" not in headers
        assert "x-uipath-traceid" not in headers

    def test_process_key_encoded(self) -> None:
        env = {"UIPATH_PROCESS_KEY": "My Agent-請-test"}
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_headers("fake-token")
        value = headers["X-UiPath-ProcessKey"]
        assert "請" not in value
        value.encode("ascii")

    def test_optional_gateway_headers(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            headers = build_uipath_headers(
                "fake-token",
                agenthub_config="config-abc",
                byo_connection_id="conn-xyz",
            )
        assert headers["X-UiPath-AgentHub-Config"] == "config-abc"
        assert headers["X-UiPath-LlmGateway-ByoIsConnectionId"] == "conn-xyz"
