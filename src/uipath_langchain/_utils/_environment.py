import os


def get_execution_folder_path() -> str | None:
    """Reads the agent's executing folder path from the runtime environment."""
    return os.environ.get("UIPATH_FOLDER_PATH")


def get_execution_folder_key() -> str | None:
    """Reads the agent's executing folder key from the runtime environment.

    Some job environments (e.g. eval serverless jobs) only expose the folder
    key, not the folder path.
    """
    return os.environ.get("UIPATH_FOLDER_KEY")


def get_default_timeout() -> float:
    return float(os.getenv("UIPATH_TIMEOUT_SECONDS", "895"))
