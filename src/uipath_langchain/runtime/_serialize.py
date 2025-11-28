from enum import Enum
from typing import Any


def serialize_output(output: Any) -> Any:
    """
    Recursively serialize an output object.

    Args:
        output: The object to serialize

    Returns:
        Dict[str, Any]: Serialized output as dictionary
    """
    if output is None:
        return {}

    # Handle Pydantic models
    if hasattr(output, "model_dump"):
        return serialize_output(output.model_dump(by_alias=True))
    elif hasattr(output, "dict"):
        return serialize_output(output.dict())
    elif hasattr(output, "to_dict"):
        return serialize_output(output.to_dict())

    # Handle dictionaries
    elif isinstance(output, dict):
        return {k: serialize_output(v) for k, v in output.items()}

    # Handle lists
    elif isinstance(output, list):
        return [serialize_output(item) for item in output]

    # Handle other iterables (convert to dict first)
    elif hasattr(output, "__iter__") and not isinstance(output, (str, bytes)):
        try:
            return serialize_output(dict(output))
        except (TypeError, ValueError):
            return output

    # Handle Enums
    elif isinstance(output, Enum):
        return output.value

    # Return primitive types as is
    return output
