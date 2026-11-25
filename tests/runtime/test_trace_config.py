import json

from openinference.instrumentation.config import REDACTED_VALUE
from openinference.semconv.trace import SpanAttributes

from uipath_langchain._tracing import UiPathTraceConfig
from uipath_langchain._tracing._trace_config import redact_masked_values

_STATE = {
    "messages": [{"content": "My email is [Email_1]"}],
    "inner_state": {
        "masked_values": {"[Email_1]": "jane@example.com", "[Person_1]": "Jane Doe"},
        "masked_tool_calls": {"guardrail:call-1": ["[Email_1]"]},
    },
}


def _mask(key: str, value: str) -> str:
    masked = UiPathTraceConfig().mask(key, value)
    assert isinstance(masked, str)
    return masked


def test_redacts_the_original_values_but_keeps_the_placeholders():
    for key in (SpanAttributes.INPUT_VALUE, SpanAttributes.OUTPUT_VALUE):
        masked = json.loads(_mask(key, json.dumps(_STATE)))

        assert masked["inner_state"]["masked_values"] == {
            "[Email_1]": REDACTED_VALUE,
            "[Person_1]": REDACTED_VALUE,
        }
        assert masked["inner_state"]["masked_tool_calls"] == {
            "guardrail:call-1": ["[Email_1]"]
        }
        assert masked["messages"] == _STATE["messages"]


def test_redacts_a_state_nested_under_another_key():
    serialized = json.dumps({"input": _STATE}, ensure_ascii=False)

    masked = _mask(SpanAttributes.INPUT_VALUE, serialized)

    assert "jane@example.com" not in masked
    assert "Jane Doe" not in masked
    assert "[Email_1]" in masked


def test_redacts_a_lazy_value():
    masked = UiPathTraceConfig().mask(
        SpanAttributes.INPUT_VALUE, lambda: json.dumps(_STATE)
    )

    assert isinstance(masked, str)
    assert "jane@example.com" not in masked


def test_leaves_other_attributes_and_states_without_masked_values_alone():
    plain = json.dumps({"inner_state": {"initial_message_count": 2}})

    assert _mask(SpanAttributes.INPUT_VALUE, plain) == plain
    assert _mask("metadata", json.dumps(_STATE)) == json.dumps(_STATE)


def test_leaves_a_masked_values_key_outside_the_inner_state_alone():
    serialized = json.dumps({"masked_values": {"a": "b"}})

    assert redact_masked_values(serialized) == serialized


def test_redacts_the_whole_attribute_when_it_is_not_json():
    masked = _mask(
        SpanAttributes.INPUT_VALUE,
        "inner_state=... 'masked_values': {'[Email_1]': 'jane@example.com'}",
    )

    assert masked == REDACTED_VALUE
