"""Tests for the static-argument tool-wrapping helpers.

``wrap_tool_with_static_args`` hides a tool's *static* parameters from the
model-facing schema and injects the configured values just before the
underlying tool runs (a per-tool, state-independent counterpart to
``StaticArgsHandler``).
"""

from typing import Any

from pydantic import BaseModel, Field
from uipath.agent.models.agent import (
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolStaticArgumentProperties,
)

from uipath_langchain.agent.tools.schema_editing import remove_fields_from_schema
from uipath_langchain.agent.tools.static_args import (
    wrap_tool_with_static_args,
    wrap_tools_with_static_args,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)


class _ToolInput(BaseModel):
    host: str
    port: int = Field(default=8080)
    api_key: str


def _recording_tool(
    argument_properties: dict[str, AgentToolArgumentProperties],
) -> tuple[StructuredToolWithArgumentProperties, dict[str, Any]]:
    """A tool whose coroutine records the kwargs it is ultimately called with."""
    received: dict[str, Any] = {}

    async def tool_fn(**kwargs: Any) -> str:
        received.update(kwargs)
        return "ok"

    tool = StructuredToolWithArgumentProperties(
        name="rec",
        description="A recording tool",
        args_schema=_ToolInput,
        coroutine=tool_fn,
        output_type=None,
        argument_properties=argument_properties,
    )
    return tool, received


def _static(value: Any) -> AgentToolStaticArgumentProperties:
    return AgentToolStaticArgumentProperties(value=value, is_sensitive=False)


def _argument(path: str) -> AgentToolArgumentArgumentProperties:
    return AgentToolArgumentArgumentProperties(argument_path=path, is_sensitive=False)


async def test_static_arg_hidden_from_schema_and_injected_on_call() -> None:
    props: dict[str, AgentToolArgumentProperties] = {"$['host']": _static("localhost")}
    tool, received = _recording_tool(props)
    wrapped = wrap_tool_with_static_args(tool)

    # 'host' is hidden from the model-facing schema; the rest remain.
    args_schema = wrapped.args_schema
    assert args_schema is not None and not isinstance(args_schema, dict)
    properties = args_schema.model_json_schema()["properties"]
    assert "host" not in properties
    assert "port" in properties
    assert "api_key" in properties

    # Invoking with only the non-static args injects the static value.
    await wrapped.ainvoke({"port": 9090, "api_key": "k"})
    assert received == {"host": "localhost", "port": 9090, "api_key": "k"}


def test_static_property_dropped_but_non_static_kept() -> None:
    props: dict[str, AgentToolArgumentProperties] = {
        "$['host']": _static("localhost"),
        "$['api_key']": _argument("key"),
    }
    tool, _ = _recording_tool(props)
    wrapped = wrap_tool_with_static_args(tool)

    kept = getattr(wrapped, "argument_properties", {})
    assert "$['host']" not in kept  # static is now baked into the wrapper
    assert "$['api_key']" in kept  # argument-variant left for StaticArgsHandler


def test_original_tool_is_not_mutated() -> None:
    props: dict[str, AgentToolArgumentProperties] = {"$['host']": _static("localhost")}
    tool, _ = _recording_tool(props)
    wrap_tool_with_static_args(tool)

    original_schema = tool.args_schema
    assert original_schema is not None and not isinstance(original_schema, dict)
    assert "host" in original_schema.model_json_schema()["properties"]
    assert "$['host']" in tool.argument_properties


def test_tool_without_static_args_passthrough() -> None:
    props: dict[str, AgentToolArgumentProperties] = {"$['host']": _argument("h")}
    tool, _ = _recording_tool(props)
    assert wrap_tool_with_static_args(tool) is tool


def test_wrap_tools_with_static_args_maps_list() -> None:
    static_props: dict[str, AgentToolArgumentProperties] = {
        "$['host']": _static("localhost")
    }
    plain_props: dict[str, AgentToolArgumentProperties] = {"$['host']": _argument("h")}
    static_tool, _ = _recording_tool(static_props)
    plain_tool, _ = _recording_tool(plain_props)

    wrapped = wrap_tools_with_static_args([static_tool, plain_tool])
    assert len(wrapped) == 2
    assert wrapped[0] is not static_tool  # static tool was wrapped
    assert wrapped[1] is plain_tool  # no static args -> returned unchanged


def test_remove_fields_from_schema_top_level() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a", "b"],
    }
    removed = remove_fields_from_schema(schema, ["$['a']"])
    assert removed == {"$['a']"}
    assert "a" not in schema["properties"]
    assert "b" in schema["properties"]
    assert schema["required"] == ["b"]


def test_remove_fields_from_schema_skips_missing_and_array_paths() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    removed = remove_fields_from_schema(schema, ["$['missing']", "$['a'][*]"])
    assert removed == set()
    assert "a" in schema["properties"]


def test_wrapped_description_lists_static_params() -> None:
    props: dict[str, AgentToolArgumentProperties] = {
        "$['host']": _static("db.internal")
    }
    tool, _ = _recording_tool(props)
    wrapped = wrap_tool_with_static_args(tool)

    # Original description is preserved, plus a line naming the hidden static
    # parameter and its value so the model can still reason about it.
    assert tool.description in wrapped.description
    assert "host" in wrapped.description
    assert "db.internal" in wrapped.description


def test_wrapped_description_redacts_sensitive_static_params() -> None:
    props: dict[str, AgentToolArgumentProperties] = {
        "$['host']": AgentToolStaticArgumentProperties(
            value="super-secret-value", is_sensitive=True
        )
    }
    tool, _ = _recording_tool(props)
    wrapped = wrap_tool_with_static_args(tool)

    assert "host" in wrapped.description  # the name is still shown
    assert "<sensitive>" in wrapped.description  # but the value is redacted
    assert "super-secret-value" not in wrapped.description


def test_same_named_tools_get_distinct_descriptions() -> None:
    """Two identically-named tools differing only by a static value (e.g. a
    send-email tool for Bob vs Alice) become distinguishable via the description."""
    bob_props: dict[str, AgentToolArgumentProperties] = {
        "$['host']": _static("bob@acme.com")
    }
    alice_props: dict[str, AgentToolArgumentProperties] = {
        "$['host']": _static("alice@acme.com")
    }
    bob, _ = _recording_tool(bob_props)
    alice, _ = _recording_tool(alice_props)

    wrapped_bob = wrap_tool_with_static_args(bob)
    wrapped_alice = wrap_tool_with_static_args(alice)

    assert wrapped_bob.name == wrapped_alice.name  # same (user-configured) name
    assert wrapped_bob.description != wrapped_alice.description
    assert "bob@acme.com" in wrapped_bob.description
    assert "alice@acme.com" in wrapped_alice.description
