import logging
import os

from uipath.platform.common.constants import ENV_FOLDER_PATH

logger = logging.getLogger(__name__)


def get_execution_folder_path() -> str | None:
    """Reads the agent's executing folder path from the runtime environment."""
    folder_path = os.environ.get(ENV_FOLDER_PATH)
    logger.info("Folder path for current execution context: %s", folder_path)

    return folder_path
