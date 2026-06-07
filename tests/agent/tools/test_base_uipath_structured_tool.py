"""Tests for BaseUiPathStructuredTool to ensure it stays in sync with StructuredTool."""

from types import CodeType

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)


def _assert_code_objects_match_except_varnames(
    base_code: CodeType, struct_code: CodeType, method_name: str
) -> None:
    """Compare two code objects ensuring they're identical except for the first parameter name (self).

    Args:
        base_code: Code object from BaseUiPathStructuredTool
        struct_code: Code object from StructuredTool
        method_name: Name of the method being compared (for error messages)
    """
    assert base_code.co_code == struct_code.co_code, (
        f"{method_name}: Bytecode mismatch (length {len(base_code.co_code)} vs {len(struct_code.co_code)})"
    )
    assert base_code.co_consts == struct_code.co_consts, (
        f"{method_name}: Constants mismatch: {base_code.co_consts} vs {struct_code.co_consts}"
    )
    assert base_code.co_names == struct_code.co_names, (
        f"{method_name}: Names mismatch: {base_code.co_names} vs {struct_code.co_names}"
    )

    base_varnames = list(base_code.co_varnames)
    struct_varnames = list(struct_code.co_varnames)

    assert len(base_varnames) == len(struct_varnames), (
        f"{method_name}: Variable count mismatch: {len(base_varnames)} vs {len(struct_varnames)}"
    )

    assert struct_varnames[0] == "self", (
        f"{method_name}: Expected 'self' in StructuredTool but got '{struct_varnames[0]}'"
    )
    assert base_varnames[0] == "__obj_internal_self__", (
        f"{method_name}: Expected '__obj_internal_self__' in BaseUiPathStructuredTool "
        f"but got '{base_varnames[0]}'"
    )

    assert base_varnames[1:] == struct_varnames[1:], (
        f"{method_name}: Variable names mismatch (excluding first): "
        f"{base_varnames[1:]} vs {struct_varnames[1:]}"
    )


def test_run_implementation_matches_structured_tool():
    """Verify that _run implementation matches StructuredTool except for the first parameter name (self).

    If this test fails and BaseUiPathStructuredTool._run has NOT been modified,
    it means the upstream langchain_core.tools.StructuredTool._run implementation
    has changed and BaseUiPathStructuredTool is now out of sync.

    Action required: Update BaseUiPathStructuredTool._run to match the new
    StructuredTool._run implementation, keeping only the 'self' -> '__obj_internal_self__' rename.
    """
    try:
        _assert_code_objects_match_except_varnames(
            BaseUiPathStructuredTool._run.__code__,
            StructuredTool._run.__code__,
            "_run",
        )
    except AssertionError as e:
        msg = (
            "\n\nIMPLEMENTATION OUT OF SYNC:\n"
            "If BaseUiPathStructuredTool._run was NOT modified, this means the upstream "
            "langchain_core.tools.StructuredTool._run has changed.\n\n"
            "Action required:\n"
            "  1. Check the new StructuredTool._run implementation\n"
            "  2. Update BaseUiPathStructuredTool._run to match\n"
            "  3. Keep ONLY the 'self' -> '__obj_internal_self__' parameter rename\n\n"
        )
        raise AssertionError(msg + str(e)) from e


def test_arun_implementation_matches_structured_tool():
    """Verify that _arun implementation matches StructuredTool except for the first parameter name (self).

    If this test fails and BaseUiPathStructuredTool._arun has NOT been modified,
    it means the upstream langchain_core.tools.StructuredTool._arun implementation
    has changed and BaseUiPathStructuredTool is now out of sync.

    Action required: Update BaseUiPathStructuredTool._arun to match the new
    StructuredTool._arun implementation, keeping only the 'self' -> '__obj_internal_self__' rename.
    """
    try:
        _assert_code_objects_match_except_varnames(
            BaseUiPathStructuredTool._arun.__code__,
            StructuredTool._arun.__code__,
            "_arun",
        )
    except AssertionError as e:
        msg = (
            "\n\nIMPLEMENTATION OUT OF SYNC:\n"
            "If BaseUiPathStructuredTool._arun was NOT modified, this means the upstream "
            "langchain_core.tools.StructuredTool._arun has changed.\n\n"
            "Action required:\n"
            "  1. Check the new StructuredTool._arun implementation\n"
            "  2. Update BaseUiPathStructuredTool._arun to match\n"
            "  3. Keep ONLY the 'self' -> '__obj_internal_self__' parameter rename\n\n"
        )
        raise AssertionError(msg + str(e)) from e


def test_function_with_self_parameter():
    """Verify that a function with 'self' parameter can be invoked without conflicts."""

    class Args(BaseModel):
        self: str
        value: int

    def my_function(self: str, value: int) -> str:
        """A function with a 'self' parameter that is not the instance reference."""
        return f"{self}:{value}"

    tool = BaseUiPathStructuredTool(
        func=my_function,
        name="test_tool",
        description="Test tool with self parameter",
        args_schema=Args,
    )

    result = tool.invoke({"self": "test", "value": 42})
    assert result == "test:42"


async def _noop_coroutine(**kwargs: object) -> dict[str, object]:
    return kwargs


def test_tool_call_schema_preserves_reserved_name_aliases():
    """The JSON schema exposed to the LLM must use the user-facing property names
    (e.g. 'schema'), not the Python-safe field names chosen by the converter
    (e.g. 'schema_'). If the LLM sees 'schema_', it emits 'schema_' and the value
    is silently dropped into the model's 'extra' bucket.
    """
    input_schema = create_model(
        {
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "json": {"type": "string"},
            },
            "required": ["schema", "json"],
        }
    )

    tool = BaseUiPathStructuredTool(
        coroutine=_noop_coroutine,
        name="validator",
        description="validator",
        args_schema=input_schema,
    )

    tool_call_schema = tool.tool_call_schema
    assert isinstance(tool_call_schema, type) and issubclass(
        tool_call_schema, BaseModel
    )
    properties = tool_call_schema.model_json_schema()["properties"]
    assert set(properties) == {"schema", "json"}


@pytest.mark.asyncio
async def test_coroutine_receives_reserved_pydantic_name_as_value():
    """A tool argument aliased to a reserved Pydantic name (e.g. 'schema') must
    reach the coroutine as the user-supplied value, not as a bound BaseModel method.
    """
    input_schema = create_model(
        {
            "title": "Input",
            "type": "object",
            "properties": {
                "schema": {"type": "string"},
                "name": {"type": "string"},
            },
        }
    )

    received: dict[str, object] = {}

    async def my_coroutine(**kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return input_schema.model_validate(kwargs).model_dump()

    tool = BaseUiPathStructuredTool(
        coroutine=my_coroutine,
        name="reserved_name_tool",
        description="Tool with a reserved-name argument",
        args_schema=input_schema,
    )

    result = await tool.ainvoke({"schema": "my_schema", "name": "alice"})

    assert received == {"schema": "my_schema", "name": "alice"}
    assert result == {"schema": "my_schema", "name": "alice"}


@pytest.mark.asyncio
async def test_coroutine_with_self_parameter():
    """Verify that a coroutine with 'self' parameter can be invoked without conflicts."""

    class Args(BaseModel):
        self: str
        value: int

    async def my_coroutine(self: str, value: int) -> str:
        """A coroutine with a 'self' parameter that is not the instance reference."""
        return f"{self}:{value}"

    tool = BaseUiPathStructuredTool(
        coroutine=my_coroutine,
        name="test_tool_async",
        description="Test async tool with self parameter",
        args_schema=Args,
    )

    result = await tool.ainvoke({"self": "async_test", "value": 99})
    assert result == "async_test:99"
