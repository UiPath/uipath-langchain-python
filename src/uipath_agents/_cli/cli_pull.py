"""Middleware for pulling agent projects from UiPath Studio Web."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from uipath._cli._utils._common import may_override_files
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli._utils._project_files import ProjectPullError, pull_project
from uipath._cli.middlewares import MiddlewareResult
from uipath.platform.common import UiPathConfig

from .constants import AGENT_ENTRYPOINT

if TYPE_CHECKING:
    from uipath._cli._utils._studio_project import StudioClient

console = ConsoleLogger()


def agents_pull_middleware(
    studio_client: "StudioClient | None",
    root: Path,
    overwrite: bool,
) -> MiddlewareResult:
    """Handle pull for agent projects from Studio Web.

    Checks if the remote Studio Web project contains agent.json and
    pulls the project files.
    """
    if studio_client is None:
        return MiddlewareResult(should_continue=True)

    return _pull_from_studio_web(studio_client, root, overwrite)


def _pull_from_studio_web(
    studio_client: "StudioClient",
    root: Path,
    overwrite: bool,
) -> MiddlewareResult:
    """Pull an agent project from Studio Web (agent.json detection)."""
    try:
        structure = asyncio.run(studio_client.get_project_structure_async())
    except Exception as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=f"Failed to retrieve project structure: {e}",
            should_include_stacktrace=True,
        )

    has_agent_json = any(f.name == AGENT_ENTRYPOINT for f in structure.files)
    if not has_agent_json:
        return MiddlewareResult(should_continue=True)

    if not overwrite:
        may_override = asyncio.run(may_override_files(studio_client, "local"))
        if not may_override:
            console.info("Operation aborted.")
            return MiddlewareResult(should_continue=False)

    project_id = UiPathConfig.project_id
    assert project_id is not None

    download_configuration: dict[str | None, Path] = {
        None: root,
    }

    console.log("Pulling UiPath project from Studio Web...")

    try:

        async def run_pull() -> None:
            async for update in pull_project(
                project_id, download_configuration, studio_client
            ):
                console.info(f"Processing: {update.file_path}")
                console.info(update.message)

        asyncio.run(run_pull())
        console.success("Project pulled successfully")
        return MiddlewareResult(should_continue=False)
    except ProjectPullError as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=f"Failed to pull UiPath project: {str(e)}",
        )
