import logging
import shutil
from pathlib import Path

from pydantic import ValidationError
from uipath.agent.models.agent import LowCodeAgentDefinition

from .constants import (
    AGENT_BUILDER_FILENAME,
    AGENT_FILENAME,
)
from .exceptions import (
    ConfigurationError,
    InputValidationError,
)

logger = logging.getLogger(__name__)


def _prepare_agent_run_files():
    """Copy all files from .agent-builder to root directory."""
    agent_builder_dir = Path(AGENT_BUILDER_FILENAME)

    if not agent_builder_dir.exists() or not agent_builder_dir.is_dir():
        logger.debug(f"Agent builder directory not found at {agent_builder_dir}")
        return

    try:
        for file_path in agent_builder_dir.iterdir():
            if file_path.is_file():
                target_path = Path.cwd() / file_path.name
                shutil.copy2(file_path, target_path)
                logger.info(f"Processed {file_path.name}")
    except Exception as e:
        logger.error(f"Failed to copy files from {agent_builder_dir}: {e}")
        raise


def load_agent_configuration(file_path: Path) -> LowCodeAgentDefinition:
    """Load and validate agent.json configuration.

    Raises:
        ConfigurationError: If file missing or has invalid structure
        InputValidationError: If validation against schema fails
    """
    if not file_path.exists():
        raise ConfigurationError(f"{AGENT_FILENAME} not found at {file_path}")

    try:
        return LowCodeAgentDefinition.model_validate_json(file_path.read_text())
    except ValidationError as e:
        raise InputValidationError(
            f"{AGENT_FILENAME} failed schema validation. Error: {e}",
            validation_errors=e.errors(),
        ) from e
    except (ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Invalid {AGENT_FILENAME} structure. Error: {e}"
        ) from e
