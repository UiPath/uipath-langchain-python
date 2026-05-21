"""Shared fixtures for agent tool tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from uipath.platform.errors import EnrichedException
from uipath.platform.orchestrator import Job


@pytest.fixture
def mock_process_invocation():
    """Pre-wire UiPath + interrupt mocks for a successful process invocation.

    Yields ``(mock_client, mock_interrupt, mock_job, mock_resumed_job)``. Tests can
    mutate ``mock_resumed_job.state`` or ``mock_client.jobs.extract_output_async.return_value``
    to switch from the default "successful, no output" scenario.
    """
    mock_job = MagicMock(spec=Job)
    mock_job.key = "job-key"
    mock_job.folder_key = "folder-key"

    mock_resumed_job = MagicMock(spec=Job)
    mock_resumed_job.state = "successful"

    with (
        patch("uipath_langchain.agent.tools.process_tool.UiPath") as mock_uipath_class,
        patch(
            "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
        ) as mock_interrupt,
    ):
        mock_client = MagicMock()
        mock_client.processes.invoke_async = AsyncMock(return_value=mock_job)
        mock_client.jobs.extract_output_async = AsyncMock(return_value=None)
        mock_uipath_class.return_value = mock_client
        mock_interrupt.return_value = mock_resumed_job

        yield mock_client, mock_interrupt, mock_job, mock_resumed_job


@pytest.fixture
def make_enriched_exception():
    """Factory fixture for creating EnrichedException with a given status code and URL."""

    def _make(
        status_code: int,
        body: str = "Bad Request",
        url: str = "https://cloud.uipath.com/test_/endpoint",
    ) -> EnrichedException:
        response = httpx.Response(
            status_code=status_code,
            content=body.encode(),
            request=httpx.Request("POST", url),
        )
        http_error = httpx.HTTPStatusError(
            message=body,
            request=response.request,
            response=response,
        )
        return EnrichedException(http_error)

    return _make
