"""Tests for the legacy converter module.

Covers the _create_dynamic_module function and PydanticUndefinedAnnotation
error wrapping in create_model.
"""

import sys
from typing import Any

import pytest
from pydantic import BaseModel

from uipath_langchain.agent.exceptions import AgentStartupError
from uipath_langchain.agent.react._legacy_converter import (
    _create_dynamic_module,
    create_model,
)

MODULE_NAME = "jsonschema_pydantic_converter._dynamic_test_legacy"


class TestCreateDynamicModule:
    def test_registered_in_sys_modules_under_the_given_name(self) -> None:
        m = _create_dynamic_module(MODULE_NAME)
        assert m.__name__ == MODULE_NAME
        assert sys.modules[MODULE_NAME] is m


class TestLegacyCreateModel:
    def test_simple_schema(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        model = create_model(schema, MODULE_NAME)
        assert issubclass(model, BaseModel)
        assert "name" in model.model_fields

    def test_dangling_ref_raises_agent_startup_error(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
        }
        with pytest.raises(AgentStartupError, match="Missing.*could not be resolved"):
            create_model(schema, MODULE_NAME)

    def test_model_module_is_dynamic(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"val": {"type": "integer"}},
        }
        model = create_model(schema, MODULE_NAME)
        assert model.__module__ == MODULE_NAME

    def test_marker_name_on_referenced_type(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"contact": {"$ref": "#/$defs/Contact"}},
            "$defs": {
                "Contact": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                }
            },
        }
        model = create_model(schema, MODULE_NAME)
        # The referenced type should carry __uipath_marker_name__
        module = sys.modules[model.__module__]
        classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, BaseModel) and v is not model
        ]
        assert any(hasattr(c, "__uipath_marker_name__") for c in classes)
