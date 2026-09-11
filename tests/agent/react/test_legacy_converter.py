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
    _DYNAMIC_MODULE_PREFIX,
    _create_dynamic_module,
    create_model,
)


class TestCreateDynamicModule:
    def test_creates_unique_modules(self) -> None:
        m1 = _create_dynamic_module()
        m2 = _create_dynamic_module()
        assert m1.__name__ != m2.__name__

    def test_registered_in_sys_modules(self) -> None:
        m = _create_dynamic_module()
        assert m.__name__ in sys.modules
        assert sys.modules[m.__name__] is m

    def test_name_prefix(self) -> None:
        m = _create_dynamic_module()
        assert m.__name__.startswith(_DYNAMIC_MODULE_PREFIX)


class TestLegacyCreateModel:
    def test_simple_schema(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        model = create_model(schema)
        assert issubclass(model, BaseModel)
        assert "name" in model.model_fields

    def test_dangling_ref_raises_agent_startup_error(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"$ref": "#/$defs/Missing"}},
        }
        with pytest.raises(AgentStartupError, match="Missing.*could not be resolved"):
            create_model(schema)

    def test_model_module_is_dynamic(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"val": {"type": "integer"}},
        }
        model = create_model(schema)
        assert model.__module__.startswith(_DYNAMIC_MODULE_PREFIX)

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
        model = create_model(schema)
        # The referenced type should carry __uipath_marker_name__
        module = sys.modules[model.__module__]
        classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type) and issubclass(v, BaseModel) and v is not model
        ]
        assert any(hasattr(c, "__uipath_marker_name__") for c in classes)
