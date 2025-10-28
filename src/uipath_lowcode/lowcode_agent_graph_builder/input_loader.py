import json
from pathlib import Path
from typing import Any, Optional, Union

from jsonschema_pydantic import jsonschema_to_pydantic  # type: ignore[import-untyped]
from pydantic import ValidationError
from uipath.agent.models.agent import LowCodeAgentDefinition

from .constants import (
    AGENT_CONFIG_FILENAME,
)
from .exceptions import (
    ConfigurationError,
    InputValidationError,
)


def load_agent_configuration(agent_json_path: Path) -> LowCodeAgentDefinition:
    """Load and validate agent.json configuration.

    Raises:
        ConfigurationError: If file missing or has invalid structure
        InputValidationError: If validation against schema fails
    """
    if not agent_json_path.exists():
        raise ConfigurationError(
            f"{AGENT_CONFIG_FILENAME} not found at {agent_json_path}"
        )

    try:
        return LowCodeAgentDefinition.model_validate_json(agent_json_path.read_text())
    except ValidationError as e:
        raise InputValidationError(
            f"{AGENT_CONFIG_FILENAME} failed schema validation. Error: {e}",
            validation_errors=e.errors(),
        ) from e
    except (ValueError, TypeError) as e:
        raise ConfigurationError(
            f"Invalid {AGENT_CONFIG_FILENAME} structure. Error: {e}"
        ) from e


def load_input_arguments(
    input_schema: dict[str, Any],
    input_data: Optional[Union[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Load and validate input arguments against schema.

    Raises:
        InputValidationError: If input doesn't match schema
        ConfigurationError: If input has invalid structure
    """
    try:
        if input_data is None or input_data == "":
            input_data = {}

        pydantic_input_model = jsonschema_to_pydantic(input_schema)

        if isinstance(input_data, str):
            parsed_data = pydantic_input_model.model_validate_json(input_data)
        else:
            parsed_data = pydantic_input_model.model_validate(input_data)

        return parsed_data.model_dump()

    except ValidationError as e:
        raise InputValidationError(
            "Agent input failed schema validation",
            validation_errors=e.errors(),
        ) from e
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"Invalid Agent input structure: {e}") from e
