from typing import Any, Optional, Union

from jsonschema_pydantic import jsonschema_to_pydantic


def validate_input_data(
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
            return {}

        pydantic_input_model = jsonschema_to_pydantic(input_schema)

        if isinstance(input_data, str):
            parsed_data = pydantic_input_model.model_validate_json(input_data)
        else:
            parsed_data = pydantic_input_model.model_validate(input_data)

        return parsed_data.model_dump()

    except Exception as e:
        raise Exception(
            "Agent input failed schema validation",
            validation_errors=e.errors(),
        ) from e
