"""Build memory search field settings from an agent's configured weights."""

from uipath.platform.memory import FieldSettings
from uipath.runtime.errors import UiPathErrorCategory

from .exceptions import AgentRuntimeError, AgentRuntimeErrorCode

_MIN_WEIGHT = 0.0
_MAX_WEIGHT = 1.0


def field_settings_for_weight(field_name: str, weight: float) -> FieldSettings:
    """Return the search settings for a field with a configured weight.

    A weight outside 0..1 is a mistake in the agent's memory configuration. Name
    the field and the value, so the error does not surface as a bare validation
    error for ``FieldSettings``.
    """
    if not _MIN_WEIGHT <= weight <= _MAX_WEIGHT:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.INVALID_MEMORY_FIELD_WEIGHT,
            title=f"Invalid memory weight for field '{field_name}'",
            detail=(
                f"The memory search weight for field '{field_name}' is {weight}. "
                f"It must be between {_MIN_WEIGHT} and {_MAX_WEIGHT}. Change the "
                "field's weight in the agent's memory settings."
            ),
            category=UiPathErrorCategory.DEPLOYMENT,
        )
    return FieldSettings(weight=weight)
