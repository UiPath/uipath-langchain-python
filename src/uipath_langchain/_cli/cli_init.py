import asyncio
import importlib.resources
import json
import os
import shutil
import uuid
from collections.abc import Generator
from enum import Enum
from typing import Any, Callable, overload

import click
from langgraph.graph.state import CompiledStateGraph
from pydantic import TypeAdapter
from uipath._cli._utils._console import ConsoleLogger
from uipath._cli._utils._parse_ast import (  # type: ignore
    generate_bindings,
    write_bindings_file,
    write_entry_points_file,
)
from uipath._cli.middlewares import MiddlewareResult
from uipath._cli.models.runtime_schema import Bindings, Entrypoint, Entrypoints

from uipath_langchain._cli._utils._schema import generate_schema_from_graph

from ._utils._graph import LangGraphConfig

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
    entrypoint: str,
    options: dict[str, Any] | None = None,
    write_config: Callable[[Any], str] | None = None,
) -> MiddlewareResult:
    """Middleware to check for langgraph.json and create uipath.json with schemas"""
    options = options or {}

    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:
        config.load_config()
        entrypoints = []
        mermaids = {}
        bindings = Bindings(
            version="2.0",
            resources=[],
        )
        for graph in config.graphs:
            if entrypoint and graph.name != entrypoint:
                continue

            try:
                loaded_graph = await graph.load_graph()
                state_graph = (
                    loaded_graph.builder
                    if isinstance(loaded_graph, CompiledStateGraph)
                    else loaded_graph
                )
                compiled_graph = state_graph.compile()
                schema_details = generate_schema_from_graph(compiled_graph)

                mermaids[graph.name] = compiled_graph.get_graph(xray=1).draw_mermaid()

                try:
                    should_infer_bindings = options.get("infer_bindings", True)
                    # Make sure the file path exists
                    if os.path.exists(graph.file_path) and should_infer_bindings:
                        bindings.resources.extend(
                            generate_bindings(graph.file_path).resources
                        )

                except Exception as e:
                    console.warning(
                        f"Warning: Could not generate bindings for {graph.file_path}: {str(e)}"
                    )

                new_entrypoint: dict[str, Any] = {
                    "filePath": graph.name,
                    "uniqueId": str(uuid.uuid4()),
                    "type": "agent",
                    "input": schema_details.schema["input"],
                    "output": schema_details.schema["output"],
                }
                entrypoints.append(new_entrypoint)

                warning_circular_deps = f" schema of graph '{graph.name}' contains circular dependencies. Some types might not be correctly inferred."
                if schema_details.has_input_circular_dependency:
                    console.warning("Input" + warning_circular_deps)
                if schema_details.has_output_circular_dependency:
                    console.warning("Output" + warning_circular_deps)

            except Exception as e:
                console.error(f"Error during graph load: {e}")
                return MiddlewareResult(
                    should_continue=False,
                    should_include_stacktrace=True,
                )
            finally:
                await graph.cleanup()

        if entrypoint and not entrypoints:
            console.error(f"Error: No graph found with name '{entrypoint}'")
            return MiddlewareResult(
                should_continue=False,
            )

        # add here default settings like {'isConversational': false}
        uipath_config: dict[str, Any] = {}

        if write_config:
            config_path = write_config(uipath_config)
        else:
            # Save the uipath.json file
            config_path = "uipath.json"
            with open(config_path, "w") as f:
                json.dump(uipath_config, f, indent=4)
        console.success(f"Created {click.style(config_path, fg='cyan')} file.")

        entry_points_path = write_entry_points_file(
            Entrypoints(
                entry_points=[
                    TypeAdapter(Entrypoint).validate_python(entry_point)
                    for entry_point in entrypoints
                ]
            )  # type: ignore
        )
        console.success(f"Created {click.style(entry_points_path, fg='cyan')} file.")

        for graph_name, mermaid_content in mermaids.items():
            mermaid_file_path = f"{graph_name}.mermaid"
            try:
                with open(mermaid_file_path, "w") as f:
                    f.write(mermaid_content)
                console.success(
                    f"Created {click.style(mermaid_file_path, fg='cyan')} file."
                )
            except Exception as write_error:
                console.error(
                    f"Error writing mermaid file for '{graph_name}': {str(write_error)}"
                )
                return MiddlewareResult(
                    should_continue=False,
                    should_include_stacktrace=True,
                )

        bindings_path = write_bindings_file(bindings)
        console.success(f"Created {click.style(bindings_path, fg='cyan')} file.")

        generate_agents_md_files(options)

        return MiddlewareResult(should_continue=False)

    except Exception as e:
        console.error(f"Error processing langgraph configuration: {str(e)}")
        return MiddlewareResult(
            should_continue=False,
            should_include_stacktrace=True,
        )


@overload
def langgraph_init_middleware(
    entrypoint: str,
) -> MiddlewareResult: ...


@overload
def langgraph_init_middleware(
    entrypoint: str,
    options: dict[str, Any],
    write_config: Callable[[Any], str],
) -> MiddlewareResult: ...


def langgraph_init_middleware(
    entrypoint: str,
    options: dict[str, Any] | None = None,
    write_config: Callable[[Any], str] | None = None,
) -> MiddlewareResult:
    """Middleware to check for langgraph.json and create uipath.json with schemas"""
    return asyncio.run(
        langgraph_init_middleware_async(entrypoint, options, write_config)
    )
