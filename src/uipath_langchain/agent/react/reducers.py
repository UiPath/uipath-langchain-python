"""Dict-like object reducers for merging state with field-specific reducers."""

from typing import Any

from pydantic import BaseModel
from uipath.platform.attachments import Attachment


def add_job_attachments(
    left: dict[str, Attachment], right: dict[str, Attachment]
) -> dict[str, Attachment]:
    """Merge attachment dictionaries, with right values taking precedence.

    This reducer function merges two dictionaries of attachments by UUID string.
    If the same UUID exists in both dictionaries, the value from 'right' takes precedence.

    Args:
        left: Existing dictionary of attachments keyed by UUID string
        right: New dictionary of attachments to merge

    Returns:
        Merged dictionary with right values overriding left values for duplicate keys
    """
    if not right:
        return left

    if not left:
        return right

    return {**left, **right}


def merge_objects(left: Any, right: Any) -> Any:
    """Merge a Pydantic model with another model or dict, with right values taking precedence.

    Applies field-specific reducers from annotation metadata when merging values.

    Args:
        left: Existing Pydantic BaseModel instance
        right: New Pydantic BaseModel instance or dict to merge

    Returns:
        New Pydantic model instance with merged values

    Raises:
        TypeError: If left is not a Pydantic BaseModel or right is not a BaseModel or dict
    """
    if not right:
        return left

    if not left:
        return right

    # validate input types
    if not isinstance(left, BaseModel):
        raise TypeError("Left object must be a Pydantic BaseModel")

    if not isinstance(right, (BaseModel, dict)):
        raise TypeError("Right object must be a Pydantic BaseModel or dict")

    model_fields = type(left).model_fields
    merged_values = {}

    for field_name in model_fields:
        merged_values[field_name] = getattr(left, field_name)

    for field_name in model_fields:
        if isinstance(right, BaseModel):
            if hasattr(right, field_name):
                right_value = getattr(right, field_name)
            else:
                continue  # field not present in right
        else:
            # right is dict
            if field_name not in right:
                continue  # field not present in right
            right_value = right[field_name]

        field_info = model_fields[field_name]
        left_value = merged_values[field_name]

        # apply reducer if defined
        if field_info.metadata and callable(field_info.metadata[0]):
            reducer_func = field_info.metadata[0]
            merged_values[field_name] = reducer_func(left_value, right_value)
        else:
            merged_values[field_name] = right_value

    # return new model instance with merged values
    return type(left)(**merged_values)
