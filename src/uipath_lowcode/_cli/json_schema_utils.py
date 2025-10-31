from typing import Any, Optional, Union

from jsonschema_pydantic import jsonschema_to_pydantic

from .exceptions import InputValidationError


def validate_json_against_json_schema(
    json_schema: dict[str, Any],
    arguments: Optional[Union[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Validate arguments against json schema."""
    try:
        if arguments is None or arguments == "":
            return {}

        pydantic_model = jsonschema_to_pydantic(json_schema)

        if isinstance(arguments, str):
            parsed_data = pydantic_model.model_validate_json(arguments)
        else:
            parsed_data = pydantic_model.model_validate(arguments)

        return parsed_data.model_dump()
    except Exception as e:
        raise InputValidationError(
            "Data failed json schema validation",
            validation_errors=e.errors(),
        ) from e
