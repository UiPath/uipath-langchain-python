import os


def get_execution_folder_path() -> str | None:
    """Reads the agent's executing folder path from the runtime environment."""
    return os.environ.get("UIPATH_FOLDER_PATH")
