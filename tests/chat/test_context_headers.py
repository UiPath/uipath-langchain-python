import os
from unittest.mock import PropertyMock, patch

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
            headers = build_uipath_headers()
        assert "Authorization" not in headers
        assert headers["x-uipath-jobkey"] == "job-123"
        assert headers["x-uipath-folderkey"] == "folder-456"
        assert headers["x-uipath-traceid"] == "trace-789"

    def test_no_context_headers_when_env_empty(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            headers = build_uipath_headers()
        assert headers == {}

    def test_partial_context_headers(self) -> None:
        env = {"UIPATH_JOB_KEY": "job-only"}
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_headers()
        assert headers["x-uipath-jobkey"] == "job-only"
        assert "x-uipath-folderkey" not in headers
        assert "x-uipath-traceid" not in headers

    def test_process_key_encoded(self) -> None:
        env = {"UIPATH_PROCESS_KEY": "My Agent-請-test"}
        with patch.dict(os.environ, env, clear=True):
            headers = build_uipath_headers()
        value = headers["x-uipath-processkey"]
        assert "請" not in value
        value.encode("ascii")

    def test_optional_gateway_headers(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            headers = build_uipath_headers(
                agenthub_config="config-abc",
                byo_connection_id="conn-xyz",
            )
        assert headers["x-uipath-agenthub-config"] == "config-abc"
        assert headers["x-uipath-llmgateway-byoisconnectionid"] == "conn-xyz"

    def test_licensing_context_header_present(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "uipath.platform.common._config.ConfigurationManager.licensing_context",
                new_callable=PropertyMock,
                return_value="robot:unattended",
            ),
        ):
            headers = build_uipath_headers()
        assert headers["x-uipath-licensing-context"] == "robot:unattended"

    def test_licensing_context_header_absent_when_none(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "uipath.platform.common._config.ConfigurationManager.licensing_context",
                new_callable=PropertyMock,
                return_value=None,
            ),
        ):
            headers = build_uipath_headers()
        assert "x-uipath-licensing-context" not in headers

