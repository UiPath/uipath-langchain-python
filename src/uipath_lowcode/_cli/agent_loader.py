from pathlib import Path

from pydantic import ValidationError
from uipath.agent.models.agent import LowCodeAgentDefinition

from .constants import (
    AGENT_FILENAME,
)
from .exceptions import (
    ConfigurationError,
    InputValidationError,
)


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
