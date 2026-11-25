"""OpenInference trace config that keeps masked PII out of agent state spans."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.semconv.trace import SpanAttributes
from opentelemetry.util.types import AttributeValue

_STATE_ATTRIBUTES = (SpanAttributes.INPUT_VALUE, SpanAttributes.OUTPUT_VALUE)
_INNER_STATE = "inner_state"
_MASKED_VALUES = "masked_values"


@dataclass(frozen=True)
class UiPathTraceConfig(TraceConfig):
    """Redacts the values a Mask guardrail replaced with placeholders.

    The placeholder map lives in the agent state so it survives checkpoints, and
    every LangGraph span serializes that state. The placeholders stay visible;
    the original values do not.
    """

    # **kwargs forwards `externalize`, which newer openinference releases pass and
    # older ones don't accept.
    def mask(
        self,
        key: str,
        value: Union[AttributeValue, Callable[[], AttributeValue]],
        **kwargs: Any,
    ) -> Optional[AttributeValue]:
        masked = super().mask(key, value, **kwargs)
        if (
            key in _STATE_ATTRIBUTES
            and isinstance(masked, str)
            and _MASKED_VALUES in masked
        ):
            return redact_masked_values(masked)
        return masked


def redact_masked_values(serialized: str) -> str:
    """Replace each ``inner_state.masked_values`` value in a JSON span attribute.

    Anything that cannot be parsed is redacted whole, since it may hold the values.
    """
    try:
        data = json.loads(serialized)
    except ValueError:
        return REDACTED_VALUE
    if not _redact(data):
        return serialized
    return json.dumps(data, ensure_ascii=False)


def _redact(node: Any) -> bool:
    changed = False
    if isinstance(node, dict):
        inner_state = node.get(_INNER_STATE)
        if isinstance(inner_state, dict) and isinstance(
            masked_values := inner_state.get(_MASKED_VALUES), dict
        ):
            inner_state[_MASKED_VALUES] = {
                placeholder: REDACTED_VALUE for placeholder in masked_values
            }
            changed = bool(masked_values)
        for child in node.values():
            changed = _redact(child) or changed
    elif isinstance(node, list):
        for child in node:
            changed = _redact(child) or changed
    return changed
