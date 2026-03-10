from typing import Any, Optional, Union

from pydantic import ValidationError
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model


def validate_json_against_json_schema(
    json_schema: dict[str, Any],
    arguments: Optional[Union[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Validate arguments against json schema.

    Args:
        json_schema: JSON schema definition
        arguments: Input data to validate (dict or JSON string)

    Returns:
        Validated data as dictionary

    Raises:
        AgentStartupError: If validation fails
    """
    try:
        if arguments is None or arguments == "":
            return {}

        pydantic_model = create_model(json_schema)

        if isinstance(arguments, str):
            parsed_data = pydantic_model.model_validate_json(arguments)
        else:
            parsed_data = pydantic_model.model_validate(arguments)

        return parsed_data.model_dump()
    except ValidationError as e:
        raise AgentStartupError(
            AgentStartupErrorCode.INPUT_VALIDATION_ERROR,
            "Input validation failed",
            f"Data failed json schema validation: {e}",
            UiPathErrorCategory.USER,
        ) from e
    except (ValueError, TypeError) as e:
        raise AgentStartupError(
            AgentStartupErrorCode.INPUT_VALIDATION_ERROR,
            "Input validation failed",
            f"Invalid input data: {e}",
            UiPathErrorCategory.USER,
        ) from e
