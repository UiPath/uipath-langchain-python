import os
from contextlib import contextmanager


def get_file_path(file_name: str) -> str:
    """
    Get the path to a mock file, checking multiple possible locations.

    Args:
        file_name: Name of the file to locate

    Returns:
        Absolute path to the file
    """
    if os.path.isfile(f"mocks/{file_name}"):
        return os.path.abspath(f"mocks/{file_name}")
    else:
        return os.path.abspath(f"tests/tools/mocks/{file_name}")


@contextmanager
def uipath_connection_mock(httpx_mock, response):
    httpx_mock.add_response(
        url="https://example.com/api/v1/Connections/connection-id",
        method="GET",
        json=response,
    )

    try:
        yield
    finally:
        pass


@contextmanager
def uipath_token_mock(httpx_mock, response):
    httpx_mock.add_response(
        url="https://example.com/api/v1/Connections/connection-id/token?tokenType=direct",
        method="GET",
        json=response,
    )

    try:
        yield
    finally:
        pass


@contextmanager
def uipath_integration_mock(httpx_mock, response):
    httpx_mock.add_response(
        url="https://example.uipath.com//v3/element/instances/0/integration",
        method="POST",
        json=response,
    )

    try:
        yield
    finally:
        pass


@contextmanager
def uipath_interrupt_mock(mocker, tool_response):
    mocker.patch(
        "uipath_langchain.tools.preconfigured.interrupt", return_value=tool_response
    )

    try:
        yield
    finally:
        pass
