import os

# Packager sentinel written into agent.json for solution-local resources,
# meaning "same folder as the deployed solution" (see SOLUTION_FOLDER_SENTINEL
# in the agents repo packager). It must never be sent as a real folder path.
SOLUTION_FOLDER_SENTINEL = "solution_folder"


def get_execution_folder_path() -> str | None:
    """Reads the agent's executing folder path from the runtime environment."""
    return os.environ.get("UIPATH_FOLDER_PATH")


def resolve_resource_folder_path(resource_folder_path: str | None) -> str | None:
    """Folder path where a bound resource lives, or None when unknown.

    Maps the packager's ``solution_folder`` sentinel (and empty values) to
    None so callers fall back to the execution folder.

    Args:
        resource_folder_path: The ``folderPath`` recorded on the resource config.

    Returns:
        The resource's folder path, or None when it is empty or the sentinel.
    """
    if not resource_folder_path or resource_folder_path == SOLUTION_FOLDER_SENTINEL:
        return None
    return resource_folder_path


def get_default_timeout() -> float:
    return float(os.getenv("UIPATH_TIMEOUT_SECONDS", "895"))
