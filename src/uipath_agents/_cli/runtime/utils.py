from typing import Any, Optional, Union

from pydantic import ValidationError
from uipath.utils.dynamic_schema import jsonschema_to_pydantic

from ..exceptions import InputValidationError


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
        InputValidationError: If validation fails
    """
    try:
        if arguments is None or arguments == "":
            return {}

        pydantic_model = jsonschema_to_pydantic(json_schema)

        if isinstance(arguments, str):
            parsed_data = pydantic_model.model_validate_json(arguments)
        else:
            parsed_data = pydantic_model.model_validate(arguments)

        return parsed_data.model_dump()
    except ValidationError as e:
        raise InputValidationError(
            "Data failed json schema validation",
            validation_errors=e.errors(),
        ) from e
    except (ValueError, TypeError) as e:
        raise InputValidationError(
            f"Invalid input data: {e}",
            validation_errors=None,
        ) from e
