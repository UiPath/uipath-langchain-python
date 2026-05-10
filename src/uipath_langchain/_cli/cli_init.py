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
        no_agents_md_override: When True, do not overwrite an existing file.

    Returns:
        A tuple of (file_name, status) where status is a FileOperationStatus,
        or None if an error occurred.
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
    target_directory: str,
    no_agents_md_override: bool,
    with_offline_docs: bool = False,
) -> Generator[tuple[str, FileOperationStatus], None, None]:
    """Generate AGENTS.md (and a CLAUDE.md shim), optionally bundling the offline docs.

    Args:
        target_directory: The directory where the files should be created.
        no_agents_md_override: When True, do not overwrite existing AGENTS.md/CLAUDE.md.
        with_offline_docs: When True, copy llms-full.txt to .uipath/ as an offline fallback.

    Yields:
        Tuple of (file_name, status) for each file operation.
    """
    result = generate_agent_md_file(
        target_directory,
        "AGENTS.md",
        "uipath_langchain._resources",
        no_agents_md_override,
    )
    if result:
        yield result

    claude_result = generate_agent_md_file(
        target_directory,
        "CLAUDE.md",
        "uipath_langchain._resources",
        no_agents_md_override,
    )
    if claude_result:
        yield claude_result

    if with_offline_docs:
        uipath_dir = os.path.join(target_directory, ".uipath")
        os.makedirs(uipath_dir, exist_ok=True)
        try:
            source = importlib.resources.files("uipath._resources").joinpath(
                "llms-full.txt"
            )
            with importlib.resources.as_file(source) as s_path:
                shutil.copy(s_path, os.path.join(uipath_dir, "llms-full.txt"))
        except (FileNotFoundError, ModuleNotFoundError):
            pass
        else:
            agents_path = os.path.join(target_directory, "AGENTS.md")
            with open(agents_path, "a", encoding="utf-8") as f:
                f.write(
                    "\n3. If neither of the above is reachable, read "
                    "`.uipath/llms-full.txt` (offline fallback bundled with this project).\n"
                )


def generate_agents_md_files(options: dict[str, Any]) -> None:
    """Generate AGENTS.md and log a summary."""
    current_directory = os.getcwd()
    no_agents_md_override = options.get("no_agents_md_override", False)
    with_offline_docs = options.get("with_offline_docs", False)

    created_files = []
    updated_files = []
    skipped_files = []

    for file_name, status in generate_specific_agents_md_files(
        current_directory, no_agents_md_override, with_offline_docs
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
