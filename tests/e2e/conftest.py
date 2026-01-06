"""
Pytest configuration and fixtures for e2e tests.

Required environment variables:
- UIPATH_CLIENT_ID: Client ID for authentication
- UIPATH_CLIENT_SECRET: Client secret for authentication
- UIPATH_URL: Base URL (e.g., https://alpha.uipath.com/org/tenant)
- UIPATH_PROJECT_ID: Project ID for evaluations (optional for run tests)

Optional (extracted from URL if not provided):
- UIPATH_ORGANIZATION_ID: Organization name/ID
- UIPATH_TENANT_ID: Tenant name/ID

These can be set in the environment or in a .env file.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Generator
from urllib.parse import urlparse

import httpx
import pytest

# Root directory of the repository
REPO_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# Available examples that can be tested
AVAILABLE_EXAMPLES = [
    "calculator",
    "calculator_same_as_agent",
    "basic",
    "basic_with_ootb_guardrails",
]

# Examples with evaluations configured
EXAMPLES_WITH_EVALS = [
    "calculator",
    "calculator_same_as_agent",
    "basic",
]


def get_env_var(name: str, required: bool = True) -> str | None:
    """Get environment variable with optional requirement check."""
    value = os.environ.get(name)
    if value:
        # Strip whitespace to handle secrets with trailing spaces/newlines
        value = value.strip()
    if required and not value:
        pytest.skip(f"Required environment variable {name} not set")
    return value


@pytest.fixture(scope="session")
def client_id() -> str:
    """Get UIPATH_CLIENT_ID from environment."""
    value = get_env_var("UIPATH_CLIENT_ID")
    assert value is not None
    return value


@pytest.fixture(scope="session")
def client_secret() -> str:
    """Get UIPATH_CLIENT_SECRET from environment."""
    value = get_env_var("UIPATH_CLIENT_SECRET")
    assert value is not None
    return value


@pytest.fixture(scope="session")
def base_url() -> str:
    """Get UIPATH_URL from environment."""
    value = get_env_var("UIPATH_URL")
    assert value is not None
    return value


@pytest.fixture(scope="session")
def project_id() -> str | None:
    """Get UIPATH_PROJECT_ID from environment (optional)."""
    return get_env_var("UIPATH_PROJECT_ID", required=False)


def _extract_org_tenant_from_url(url: str) -> tuple[str, str]:
    """Extract organization and tenant from UiPath URL.

    URL format: https://alpha.uipath.com/org/tenant/
    Returns: (organization, tenant)
    """
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    elif len(path_parts) == 1:
        return path_parts[0], "DefaultTenant"
    return "", ""


@pytest.fixture(scope="session")
def organization_id(base_url: str) -> str:
    """Get UIPATH_ORGANIZATION_ID from environment or extract from URL."""
    value = get_env_var("UIPATH_ORGANIZATION_ID", required=False)
    if value:
        return value
    # Extract from URL
    org, _ = _extract_org_tenant_from_url(base_url)
    if not org:
        pytest.skip("Could not determine organization from URL or environment")
    return org


@pytest.fixture(scope="session")
def tenant_id(base_url: str) -> str:
    """Get UIPATH_TENANT_ID from environment or extract from URL."""
    value = get_env_var("UIPATH_TENANT_ID", required=False)
    if value:
        return value
    # Extract from URL
    _, tenant = _extract_org_tenant_from_url(base_url)
    if not tenant:
        pytest.skip("Could not determine tenant from URL or environment")
    return tenant


@pytest.fixture(scope="session")
def auth_env(
    client_id: str,
    client_secret: str,
    base_url: str,
    organization_id: str,
    tenant_id: str,
    project_id: str | None,
) -> dict[str, str]:
    """
    Create environment dict with authentication variables.

    This allows running uipath commands with client credentials
    without needing to run `uipath auth` interactively.
    """
    env = os.environ.copy()
    env["UIPATH_CLIENT_ID"] = client_id
    env["UIPATH_CLIENT_SECRET"] = client_secret
    env["UIPATH_URL"] = base_url
    env["UIPATH_ORGANIZATION_ID"] = organization_id
    env["UIPATH_TENANT_ID"] = tenant_id
    if project_id:
        env["UIPATH_PROJECT_ID"] = project_id
    return env


def _get_uipath_executable() -> str:
    """Find the uipath executable."""
    # First try to find it in PATH or venv
    uipath_path = shutil.which("uipath")
    if uipath_path:
        return uipath_path

    # Try the venv bin directory
    venv_uipath = REPO_ROOT / ".venv" / "bin" / "uipath"
    if venv_uipath.exists():
        return str(venv_uipath)

    # Fallback to just "uipath" and hope it's in PATH
    return "uipath"


def _get_access_token(base_url: str, client_id: str, client_secret: str) -> str:
    """Get access token using client credentials flow.

    Args:
        base_url: UiPath base URL (e.g., https://alpha.uipath.com/org/tenant)
        client_id: Client ID for authentication
        client_secret: Client secret for authentication

    Returns:
        Access token string
    """
    # Extract the domain from base_url
    parsed = urlparse(base_url)
    token_url = f"{parsed.scheme}://{parsed.netloc}/identity_/connect/token"

    # Request token with scopes for execution and reporting
    # OR.Execution - Required for running agents
    # OR.Users.Read - Required for getting user info for reporting
    # OR.Folders.Read - Required for accessing personal workspace for reporting
    response = httpx.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "OR.Execution OR.Users.Read OR.Folders.Read",
        },
    )

    if response.status_code != 200:
        pytest.fail(f"Failed to get access token: {response.text}")

    return response.json()["access_token"]


@pytest.fixture(scope="session")
def authenticated_session(
    auth_env: dict[str, str],
) -> Generator[dict[str, str], None, None]:
    """
    Authenticate once per session and yield the environment.

    Gets an access token using client credentials and adds it to the environment.
    """
    # Get access token directly using client credentials
    access_token = _get_access_token(
        auth_env["UIPATH_URL"],
        auth_env["UIPATH_CLIENT_ID"],
        auth_env["UIPATH_CLIENT_SECRET"],
    )

    # Add access token to environment
    auth_env["UIPATH_ACCESS_TOKEN"] = access_token

    yield auth_env


@pytest.fixture
def examples_dir() -> Path:
    """Return the examples directory path."""
    return EXAMPLES_DIR


@pytest.fixture(params=AVAILABLE_EXAMPLES)
def example_name(request) -> str:
    """Parametrized fixture for all available examples."""
    return request.param


@pytest.fixture(params=EXAMPLES_WITH_EVALS)
def example_with_evals(request) -> str:
    """Parametrized fixture for examples that have evaluations."""
    return request.param


def run_uipath_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """
    Run a uipath CLI command.

    Args:
        command: Command arguments (e.g., ["run", "agent.json"])
        cwd: Working directory
        env: Environment variables
        timeout: Command timeout in seconds

    Returns:
        CompletedProcess with stdout, stderr, and returncode
    """
    uipath_cmd = _get_uipath_executable()
    full_command = [uipath_cmd] + command

    # Print command being executed for debugging
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Running: {' '.join(full_command)}", file=sys.stderr)
    print(f"Working directory: {cwd}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    result = subprocess.run(
        full_command,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=timeout,
    )

    # Print output for debugging
    if result.stdout:
        print(f"\n--- STDOUT ---\n{result.stdout}", file=sys.stderr)
    if result.stderr:
        print(f"\n--- STDERR ---\n{result.stderr}", file=sys.stderr)
    print(f"\n--- Return code: {result.returncode} ---", file=sys.stderr)

    return result
