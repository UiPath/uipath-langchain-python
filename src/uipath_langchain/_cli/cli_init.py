import asyncio
import importlib.resources
import os
import shutil
from collections.abc import Generator
from enum import Enum
from typing import Any

import click
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli.middlewares import MiddlewareResult

from uipath_langchain.runtime.config import LangGraphConfig

console = ConsoleLogger()


class FileOperationStatus(str, Enum):
    """Status of a file operation."""

    CREATED = "created"
    UPDATED = "updated"
    SKIPPED = "skipped"


def generate_agent_md_file(
    target_directory: str,
    file_name: str,
    resource_name: str,
    no_agents_md_override: bool,
) -> tuple[str, FileOperationStatus] | None:
    """Generate an agent-specific file from the packaged resource.

    Args:
        target_directory: The directory where the file should be created.
        file_name: The name of the file should be created.
        resource_name: The name of the resource folder where should be the file.
        no_agents_md_override: Whether to override existing files.

    Returns:
        A tuple of (file_name, status) where status is a FileOperationStatus:
        - CREATED: File was created
        - UPDATED: File was overwritten
        - SKIPPED: File exists and no_agents_md_override is True
        Returns None if an error occurred.
    """
    target_path = os.path.join(target_directory, file_name)
    will_override = os.path.exists(target_path)

    if will_override and no_agents_md_override:
        return file_name, FileOperationStatus.SKIPPED
    try:
        source_path = importlib.resources.files(resource_name).joinpath(file_name)

        with importlib.resources.as_file(source_path) as s_path:
            shutil.copy(s_path, target_path)

        return (
            file_name,
            FileOperationStatus.UPDATED
            if will_override
            else FileOperationStatus.CREATED,
        )

    except Exception as e:
        console.warning(f"Could not create {file_name}: {e}")
        return None


def generate_specific_agents_md_files(
    target_directory: str, no_agents_md_override: bool
) -> Generator[tuple[str, FileOperationStatus], None, None]:
    """Generate agent-specific files from the packaged resource.

    Args:
        target_directory: The directory where the files should be created.
        no_agents_md_override: Whether to override existing files.

    Yields:
        Tuple of (file_name, status) for each file operation, where status is a FileOperationStatus:
        - CREATED: File was created
        - UPDATED: File was overwritten
        - SKIPPED: File exists and was not overwritten
    """
    agent_dir = os.path.join(target_directory, ".agent")
    os.makedirs(agent_dir, exist_ok=True)

    file_configs = [
        (target_directory, "CLAUDE.md", "uipath._resources"),
        (agent_dir, "CLI_REFERENCE.md", "uipath._resources"),
        (agent_dir, "SDK_REFERENCE.md", "uipath._resources"),
        (target_directory, "AGENTS.md", "uipath_langchain._resources"),
        (agent_dir, "REQUIRED_STRUCTURE.md", "uipath_langchain._resources"),
    ]

    for directory, file_name, resource_name in file_configs:
        result = generate_agent_md_file(
            directory, file_name, resource_name, no_agents_md_override
        )
        if result:
            yield result


def generate_agents_md_files(options: dict[str, Any]) -> None:
    """Generate agent MD files and log categorized summary.

    Args:
        options: Options dictionary
    """
    current_directory = os.getcwd()
    no_agents_md_override = options.get("no_agents_md_override", False)

    created_files = []
    updated_files = []
    skipped_files = []

    for file_name, status in generate_specific_agents_md_files(
        current_directory, no_agents_md_override
    ):
        if status == FileOperationStatus.CREATED:
            created_files.append(file_name)
        elif status == FileOperationStatus.UPDATED:
            updated_files.append(file_name)
        elif status == FileOperationStatus.SKIPPED:
            skipped_files.append(file_name)

    if created_files:
        files_str = ", ".join(click.style(f, fg="cyan") for f in created_files)
        console.success(f"Created: {files_str}")

    if updated_files:
        files_str = ", ".join(click.style(f, fg="cyan") for f in updated_files)
        console.success(f"Updated: {files_str}")

    if skipped_files:
        files_str = ", ".join(click.style(f, fg="yellow") for f in skipped_files)
        console.info(f"Skipped (already exist): {files_str}")


async def langgraph_init_middleware_async(
    options: dict[str, Any] | None = None,
) -> MiddlewareResult:
    """Middleware to check for langgraph.json and create uipath.json with schemas"""
    options = options or {}

    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        generate_agents_md_files(options)

        return MiddlewareResult(should_continue=False)

    except Exception as e:
        console.error(f"Error processing langgraph configuration: {str(e)}")
        return MiddlewareResult(
            should_continue=False,
            should_include_stacktrace=True,
        )


def langgraph_init_middleware(
    options: dict[str, Any] | None = None,
) -> MiddlewareResult:
    """Middleware to check for langgraph.json and create uipath.json with schemas"""
    return asyncio.run(langgraph_init_middleware_async(options))
