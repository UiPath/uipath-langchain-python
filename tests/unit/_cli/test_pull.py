"""Tests for agents pull middleware."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pytest_httpx import HTTPXMock
from uipath._cli._utils._studio_project import StudioClient
from uipath._cli.middlewares import Middlewares

from uipath_agents._cli.cli_pull import agents_pull_middleware

BASE_URL = "https://cloud.uipath.com/organization"
PROJECT_ID = "test-project-id"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UIPATH_URL", f"{BASE_URL}/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "mock_token")
    monkeypatch.setenv("UIPATH_PROJECT_ID", PROJECT_ID)


def _agent_project_structure() -> dict[str, object]:
    """Remote project structure containing agent.json (agents project)."""
    return {
        "id": "root",
        "name": "root",
        "folders": [],
        "files": [
            {
                "id": "agent-json-id",
                "name": "agent.json",
                "isMain": False,
                "fileType": "1",
                "isEntryPoint": False,
                "ignoredFromPublish": False,
            },
            {
                "id": "bindings-json-id",
                "name": "bindings.json",
                "isMain": False,
                "fileType": "1",
                "isEntryPoint": False,
                "ignoredFromPublish": False,
            },
        ],
        "folderType": "0",
    }


def _coded_project_structure() -> dict[str, object]:
    """Remote project structure containing pyproject.toml (coded project)."""
    return {
        "id": "root",
        "name": "root",
        "folders": [],
        "files": [
            {
                "id": "main-py-id",
                "name": "main.py",
                "isMain": True,
                "fileType": "1",
                "isEntryPoint": True,
                "ignoredFromPublish": False,
            },
            {
                "id": "pyproject-toml-id",
                "name": "pyproject.toml",
                "isMain": False,
                "fileType": "1",
                "isEntryPoint": False,
                "ignoredFromPublish": False,
            },
        ],
        "folderType": "0",
    }


# --- Studio Web Pull Tests ---


class TestAgentsPullMiddleware:
    """Tests for agents_pull_middleware function (Studio Web mode)."""

    def test_passes_through_for_coded_project(self, httpx_mock: HTTPXMock) -> None:
        """Middleware returns should_continue=True when project has no agent.json."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/Structure",
            json=_coded_project_structure(),
        )

        studio_client = StudioClient(project_id=PROJECT_ID)
        result = agents_pull_middleware(studio_client, Path("."), overwrite=True)

        assert result.should_continue is True
        assert result.error_message is None

    def test_pulls_agents_project(self, tmp_path: Path, httpx_mock: HTTPXMock) -> None:
        """Middleware pulls files and returns should_continue=False for agents project."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/Structure",
            json=_agent_project_structure(),
        )

        agent_config = {
            "id": "test-agent",
            "name": "Test Agent",
            "messages": [],
            "settings": {"model": "gpt-4", "engine": "basic-v2"},
        }
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/File/agent-json-id",
            content=json.dumps(agent_config).encode(),
        )

        bindings = {"version": "2.0", "resources": []}
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/File/bindings-json-id",
            content=json.dumps(bindings).encode(),
        )

        studio_client = StudioClient(project_id=PROJECT_ID)
        result = agents_pull_middleware(studio_client, tmp_path, overwrite=True)

        assert result.should_continue is False
        assert result.error_message is None

        agent_json_path = tmp_path / "agent.json"
        assert agent_json_path.exists()
        with open(agent_json_path, "r") as f:
            assert json.load(f) == agent_config

        bindings_path = tmp_path / "bindings.json"
        assert bindings_path.exists()
        with open(bindings_path, "r") as f:
            assert json.load(f) == bindings

    def test_pulls_agents_project_with_existing_files(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Middleware updates files that differ from remote."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/Structure",
            json=_agent_project_structure(),
        )

        remote_agent_config = {
            "id": "test-agent",
            "name": "Updated Agent",
            "messages": [],
            "settings": {"model": "gpt-4", "engine": "basic-v2"},
        }
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/File/agent-json-id",
            content=json.dumps(remote_agent_config).encode(),
        )

        bindings = {"version": "2.0", "resources": []}
        httpx_mock.add_response(
            method="GET",
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/File/bindings-json-id",
            content=json.dumps(bindings).encode(),
        )

        local_agent_config = {"id": "test-agent", "name": "Old Agent"}
        agent_json_path = tmp_path / "agent.json"
        agent_json_path.write_text(json.dumps(local_agent_config))

        studio_client = StudioClient(project_id=PROJECT_ID)
        result = agents_pull_middleware(studio_client, tmp_path, overwrite=True)

        assert result.should_continue is False

        with open(agent_json_path, "r") as f:
            assert json.load(f) == remote_agent_config

    def test_aborts_when_user_declines_override(self, httpx_mock: HTTPXMock) -> None:
        """Middleware aborts when may_override_files returns False."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/Structure",
            json=_agent_project_structure(),
        )

        studio_client = StudioClient(project_id=PROJECT_ID)

        with patch(
            "uipath_agents._cli.cli_pull.may_override_files",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = agents_pull_middleware(studio_client, Path("."), overwrite=False)

        assert result.should_continue is False
        assert result.error_message is None

    def test_returns_error_on_structure_failure(self, httpx_mock: HTTPXMock) -> None:
        """Middleware returns error when project structure retrieval fails."""
        httpx_mock.add_response(
            url=f"{BASE_URL}/studio_/backend/api/Project/{PROJECT_ID}/FileOperations/Structure",
            status_code=500,
            json={"message": "Internal Server Error"},
        )

        studio_client = StudioClient(project_id=PROJECT_ID)
        result = agents_pull_middleware(studio_client, Path("."), overwrite=True)

        assert result.should_continue is False
        assert result.error_message is not None

    def test_passes_through_when_studio_client_is_none(self) -> None:
        """Middleware returns should_continue=True when no studio_client."""
        result = agents_pull_middleware(None, Path("."), overwrite=True)

        assert result.should_continue is True
        assert result.error_message is None


# --- Middleware Registration Tests ---


class TestAgentsPullMiddlewareRegistration:
    """Tests for middleware registration via entry point."""

    def test_middleware_registers_for_pull(self) -> None:
        """Verify agents_pull_middleware can be registered for 'pull' command."""
        Middlewares.clear("pull")
        Middlewares.register("pull", agents_pull_middleware)

        middlewares = Middlewares.get("pull")
        assert agents_pull_middleware in middlewares

        Middlewares.clear("pull")
