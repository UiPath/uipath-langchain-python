"""Cross-type matrix for the static-argument tool-wrapping helpers.

Exercises ``wrap_tool_with_static_args`` / ``wrap_tools_with_static_args`` across
the parameter shapes a configured tool can take:

* static parameters (hidden from the model, injected at call time)
* parameters that come from agent input (argument-variant: left untouched)
* tools where *every* parameter is static (no model-facing input remains)
* tools with no parameters
* tools where no parameter is static
* tools with very large descriptions
* tools with a large number of parameters

...crossed with the concrete tool kinds used in production: MCP tools (dict
JSON-schema), agent-resource tools (pydantic schema, optional execution
wrapper), and A2A tools (wrapper-bearing, no static args).
"""

from typing import Any, Callable

import pytest
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from pydantic import create_model as make_model
from uipath.agent.models.agent import (
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolStaticArgumentProperties,
)

from uipath_langchain.agent.tools.a2a.a2a_tool import (
    A2aStructuredToolWithWrapper,
    A2aToolInput,
)
from uipath_langchain.agent.tools.extraction_tool import StructuredToolWithWrapper
from uipath_langchain.agent.tools.static_args import (
    wrap_tool_with_static_args,
    wrap_tools_with_static_args,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


def _p(name: str) -> str:
    """Top-level jsonpath for a parameter name, e.g. ``host`` -> ``$['host']``."""
    return f"$['{name}']"


def _static(value: Any) -> AgentToolStaticArgumentProperties:
    return AgentToolStaticArgumentProperties(value=value, is_sensitive=False)


def _sensitive(value: Any) -> AgentToolStaticArgumentProperties:
    return AgentToolStaticArgumentProperties(value=value, is_sensitive=True)


def _argument(path: str) -> AgentToolArgumentArgumentProperties:
    return AgentToolArgumentArgumentProperties(argument_path=path, is_sensitive=False)


# --- tool builders, one per production tool kind -------------------------------
#
# Each builder shares the same signature so scenarios can be parametrized over
# them. They return ``(tool, received)`` where ``received`` records the kwargs
# the underlying callable is ultimately invoked with.

ToolBuilder = Callable[..., tuple[BaseTool, dict[str, Any]]]


def _dict_schema(fields: dict[str, str]) -> dict[str, Any]:
    """A raw JSON schema dict (the shape MCP tools carry)."""
    return {
        "type": "object",
        "properties": {name: {"type": typ} for name, typ in fields.items()},
        "required": list(fields),
    }


def _model_schema(fields: dict[str, str]) -> type[BaseModel]:
    """A pydantic model (the shape most agent-resource tools carry)."""
    field_defs: dict[str, Any] = {
        name: (_JSON_TO_PY[typ], ...) for name, typ in fields.items()
    }
    return make_model("DynamicInput", **field_defs)


def _mcp_tool(
    fields: dict[str, str],
    argument_properties: dict[str, AgentToolArgumentProperties],
    description: str = "An MCP tool",
    awrapper: Any | None = None,
) -> tuple[StructuredToolWithArgumentProperties, dict[str, Any]]:
    received: dict[str, Any] = {}

    async def tool_fn(**kwargs: Any) -> str:
        received.update(kwargs)
        return "ok"

    tool = StructuredToolWithArgumentProperties(
        name="mcp_tool",
        description=description,
        args_schema=_dict_schema(fields),  # MCP tools use raw dict schemas
        coroutine=tool_fn,
        output_type=Any,
        metadata={"tool_type": "mcp", "display_name": "mcp_tool", "slug": "srv"},
        argument_properties=argument_properties,
    )
    if awrapper is not None:
        tool.set_tool_wrappers(awrapper=awrapper)
    return tool, received


def _resource_tool(
    fields: dict[str, str],
    argument_properties: dict[str, AgentToolArgumentProperties],
    description: str = "An integration tool",
    awrapper: Any | None = None,
) -> tuple[StructuredToolWithArgumentProperties, dict[str, Any]]:
    received: dict[str, Any] = {}

    async def tool_fn(**kwargs: Any) -> str:
        received.update(kwargs)
        return "ok"

    tool = StructuredToolWithArgumentProperties(
        name="resource_tool",
        description=description,
        args_schema=_model_schema(fields),  # resource tools use pydantic models
        coroutine=tool_fn,
        output_type=None,
        metadata={"tool_type": "integration", "display_name": "resource_tool"},
        argument_properties=argument_properties,
    )
    if awrapper is not None:
        tool.set_tool_wrappers(awrapper=awrapper)
    return tool, received


BUILDERS = [
    pytest.param(_mcp_tool, id="mcp"),
    pytest.param(_resource_tool, id="resource"),
]


def _schema_props(tool: BaseTool) -> list[str]:
    """Model-facing property names, for either a dict (MCP) or pydantic schema."""
    args_schema = tool.args_schema
    assert args_schema is not None
    if isinstance(args_schema, dict):
        return list(args_schema.get("properties", {}))
    return list(args_schema.model_json_schema().get("properties", {}))


# --- scenario × tool-kind matrix -----------------------------------------------


@pytest.mark.parametrize("build", BUILDERS)
async def test_static_parameters(build: ToolBuilder) -> None:
    """A static parameter is hidden from the schema, surfaced in the
    description, and injected at call time; non-static params are untouched."""
    fields = {"host": "string", "port": "integer", "api_key": "string"}
    tool, received = build(fields, {_p("host"): _static("localhost")})

    wrapped = wrap_tool_with_static_args(tool)

    assert "host" not in _schema_props(wrapped)
    assert "port" in _schema_props(wrapped)
    assert "api_key" in _schema_props(wrapped)
    assert "host: localhost" in wrapped.description
    assert wrapped.metadata == tool.metadata  # metadata preserved

    await wrapped.ainvoke({"port": 9090, "api_key": "k"})
    assert received == {"host": "localhost", "port": 9090, "api_key": "k"}


@pytest.mark.parametrize("build", BUILDERS)
async def test_static_and_input_parameters(build: ToolBuilder) -> None:
    """Static params are baked in; input (argument-variant) params and plain
    params stay model-facing, and input params remain in argument_properties."""
    fields = {"provider": "string", "query": "string", "extra": "string"}
    tool, received = build(
        fields,
        {_p("provider"): _static("svc"), _p("query"): _argument("$.q")},
    )

    wrapped = wrap_tool_with_static_args(tool)

    # provider (static) hidden + baked out; query (input) + extra (plain) remain.
    assert "provider" not in _schema_props(wrapped)
    assert "query" in _schema_props(wrapped)
    assert "extra" in _schema_props(wrapped)

    kept = getattr(wrapped, "argument_properties", {})
    assert _p("provider") not in kept  # static baked into the wrapper
    assert _p("query") in kept  # input variant left for StaticArgsHandler

    await wrapped.ainvoke({"query": "hi", "extra": "e"})
    assert received == {"provider": "svc", "query": "hi", "extra": "e"}


@pytest.mark.parametrize("build", BUILDERS)
async def test_all_parameters_static_no_input(build: ToolBuilder) -> None:
    """When every parameter is static the model-facing schema is empty and the
    tool can be invoked with no arguments at all."""
    fields = {"a": "string", "b": "integer"}
    tool, received = build(fields, {_p("a"): _static("x"), _p("b"): _static(7)})

    wrapped = wrap_tool_with_static_args(tool)

    assert _schema_props(wrapped) == []  # nothing left for the model to fill
    assert getattr(wrapped, "argument_properties", {}) == {}  # all baked in
    assert "a: x" in wrapped.description
    assert "b: 7" in wrapped.description

    await wrapped.ainvoke({})
    assert received == {"a": "x", "b": 7}


@pytest.mark.parametrize("build", BUILDERS)
async def test_no_static_parameters_passthrough(build: ToolBuilder) -> None:
    """A tool whose only properties come from input (no static) is returned
    unchanged (identity), so StaticArgsHandler keeps owning it."""
    tool, _ = build({"query": "string"}, {_p("query"): _argument("$.q")})
    assert wrap_tool_with_static_args(tool) is tool


@pytest.mark.parametrize("build", BUILDERS)
async def test_no_parameters_passthrough(build: ToolBuilder) -> None:
    """A parameterless tool (no argument_properties) is returned unchanged."""
    tool, _ = build({}, {})
    assert wrap_tool_with_static_args(tool) is tool


@pytest.mark.parametrize("build", BUILDERS)
async def test_large_description_is_preserved(build: ToolBuilder) -> None:
    """A large description is preserved verbatim, with the static lines appended
    (the description itself is never truncated)."""
    big = "Lorem ipsum dolor sit amet. " * 400  # ~11k chars
    tool, _ = build(
        {"host": "string", "port": "integer"},
        {_p("host"): _static("localhost")},
        description=big,
    )

    wrapped = wrap_tool_with_static_args(tool)

    assert big in wrapped.description  # original kept in full, not truncated
    assert "host: localhost" in wrapped.description
    assert len(wrapped.description) > len(big)


@pytest.mark.parametrize("build", BUILDERS)
async def test_large_static_value_truncated_in_description_only(
    build: ToolBuilder,
) -> None:
    """A long static value is truncated in the description (a readability cap)
    but injected in full at call time."""
    long_value = "v" * 500
    tool, received = build(
        {"token": "string", "port": "integer"},
        {_p("token"): _static(long_value)},
    )

    wrapped = wrap_tool_with_static_args(tool)

    assert "..." in wrapped.description  # value truncated for display
    assert long_value not in wrapped.description  # full value not shown
    assert long_value[:200] in wrapped.description  # the cap prefix is shown

    await wrapped.ainvoke({"port": 1})
    assert received["token"] == long_value  # ...but injected in full


@pytest.mark.parametrize("build", BUILDERS)
async def test_sensitive_static_value_redacted_in_description(
    build: ToolBuilder,
) -> None:
    """A sensitive static value is redacted in the description but injected."""
    tool, received = build(
        {"api_key": "string", "port": "integer"},
        {_p("api_key"): _sensitive("super-secret")},
    )

    wrapped = wrap_tool_with_static_args(tool)

    assert "api_key: <sensitive>" in wrapped.description
    assert "super-secret" not in wrapped.description

    await wrapped.ainvoke({"port": 1})
    assert received["api_key"] == "super-secret"


@pytest.mark.parametrize("build", BUILDERS)
async def test_large_number_of_parameters(build: ToolBuilder) -> None:
    """With many parameters, every static one is hidden+injected and every
    non-static one stays model-facing."""
    static_fields = {f"s{i}": "string" for i in range(20)}
    input_fields = {f"in{i}": "string" for i in range(20)}
    fields = {**static_fields, **input_fields}
    props: dict[str, AgentToolArgumentProperties] = {
        _p(name): _static(f"val-{name}") for name in static_fields
    }
    props.update({_p(name): _argument(f"$.{name}") for name in input_fields})

    tool, received = build(fields, props)
    wrapped = wrap_tool_with_static_args(tool)

    remaining = set(_schema_props(wrapped))
    assert remaining == set(input_fields)  # only non-static remain
    assert not (remaining & set(static_fields))

    call_args = {name: f"got-{name}" for name in input_fields}
    await wrapped.ainvoke(call_args)

    assert len(received) == len(fields)
    for name in static_fields:
        assert received[name] == f"val-{name}"
    for name in input_fields:
        assert received[name] == f"got-{name}"


@pytest.mark.parametrize("build", BUILDERS)
async def test_execution_wrapper_preserved_through_static_wrapping(
    build: ToolBuilder,
) -> None:
    """Tools carry a graph-execution wrapper (set_tool_wrappers); static
    wrapping must not drop it."""

    async def _awrapper(tool: BaseTool, call: Any, state: Any) -> None:
        return None

    tool, received = build(
        {"host": "string", "port": "integer"},
        {_p("host"): _static("localhost")},
        awrapper=_awrapper,
    )

    wrapped = wrap_tool_with_static_args(tool)

    assert wrapped is not tool
    assert getattr(wrapped, "awrapper", None) is _awrapper  # survives model_copy
    await wrapped.ainvoke({"port": 1})
    assert received["host"] == "localhost"


@pytest.mark.parametrize("build", BUILDERS)
async def test_original_tool_not_mutated(build: ToolBuilder) -> None:
    """The source tool keeps its full schema, description and properties."""
    fields = {"host": "string", "port": "integer"}
    tool, _ = build(fields, {_p("host"): _static("localhost")}, description="orig")

    wrap_tool_with_static_args(tool)

    assert set(_schema_props(tool)) == {"host", "port"}
    assert tool.description == "orig"
    assert _p("host") in getattr(tool, "argument_properties", {})


# --- A2A and other wrapper-only / no-static-args tool kinds --------------------


def _a2a_tool() -> tuple[A2aStructuredToolWithWrapper, Callable[..., Any]]:
    async def _send(message: str) -> str:
        return "ok"

    async def _awrapper(tool: BaseTool, call: Any, state: Any) -> None:
        return None

    tool = A2aStructuredToolWithWrapper(
        name="remote_agent",
        description="A remote A2A agent",
        coroutine=_send,
        args_schema=A2aToolInput,
        metadata={"tool_type": "a2a"},
    )
    tool.set_tool_wrappers(awrapper=_awrapper)
    return tool, _awrapper


async def test_a2a_tool_passthrough_preserves_wrapper() -> None:
    """A2A tools carry no static argument_properties, so they pass through
    wrapping unchanged with their execution wrapper intact."""
    tool, awrapper = _a2a_tool()

    # A2A tools genuinely have no argument_properties.
    assert getattr(tool, "argument_properties", None) is None

    wrapped = wrap_tools_with_static_args([tool])
    assert wrapped[0] is tool  # identity passthrough
    assert wrapped[0].awrapper is awrapper


async def test_wrapper_only_resource_tool_passthrough() -> None:
    """Resource tools that use the execution-wrapper class but carry no static
    args (e.g. IXP extraction / escalation) pass through unchanged."""

    class _In(BaseModel):
        doc_id: str

    async def _fn(**kwargs: Any) -> str:
        return "ok"

    tool = StructuredToolWithWrapper(
        name="extraction",
        description="extract",
        args_schema=_In,
        coroutine=_fn,
        output_type=None,
        metadata={"tool_type": "ixp_extraction"},
    )
    assert wrap_tool_with_static_args(tool) is tool


async def test_plain_structured_tool_passthrough() -> None:
    """A plain LangChain StructuredTool (e.g. a client-side tool) has no
    argument_properties and is passed through unchanged."""

    class _In(BaseModel):
        value: str

    async def _fn(**kwargs: Any) -> str:
        return "ok"

    tool = StructuredTool(
        name="client_side",
        description="client side",
        args_schema=_In,
        coroutine=_fn,
    )
    assert wrap_tool_with_static_args(tool) is tool


async def test_wrap_tools_with_static_args_mixed_tool_kinds() -> None:
    """A heterogeneous list: static tools get wrapped, everything else passes
    through, and order/length are preserved."""
    static_mcp, _ = _mcp_tool({"host": "string"}, {_p("host"): _static("h")})
    static_resource, _ = _resource_tool(
        {"provider": "string", "q": "string"}, {_p("provider"): _static("svc")}
    )
    input_only, _ = _resource_tool({"q": "string"}, {_p("q"): _argument("$.q")})
    a2a, _ = _a2a_tool()

    tools: list[BaseTool] = [static_mcp, static_resource, input_only, a2a]
    wrapped = wrap_tools_with_static_args(tools)

    assert len(wrapped) == 4
    assert wrapped[0] is not static_mcp  # static -> wrapped
    assert wrapped[1] is not static_resource  # static -> wrapped
    assert wrapped[2] is input_only  # input-only -> passthrough
    assert wrapped[3] is a2a  # a2a -> passthrough
