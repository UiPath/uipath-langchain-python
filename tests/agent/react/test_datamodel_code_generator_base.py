"""Tests for UiPathDatamodelCodeGeneratorBaseModel.__getattr__ alias resolution."""

import pytest
from pydantic import Field

from uipath_langchain.agent.react._datamodel_code_generator_base import (
    UiPathDatamodelCodeGeneratorBaseModel,
)


class _ModelWithAlias(UiPathDatamodelCodeGeneratorBaseModel):
    """A model where a JSON property name is not a valid Python identifier."""

    content_type: str = Field(alias="Content-Type")
    x_custom: int = Field(default=0, alias="X-Custom")
    normal: str = "default"


class TestGetattr:
    def test_alias_resolves_to_field(self) -> None:
        m = _ModelWithAlias.model_validate(
            {"Content-Type": "application/json", "X-Custom": 42}
        )
        # Access via alias (the JSON property name) must work.
        assert getattr(m, "Content-Type") == "application/json"
        assert getattr(m, "X-Custom") == 42

    def test_real_field_name_still_works(self) -> None:
        m = _ModelWithAlias.model_validate({"Content-Type": "text/html"})
        assert m.content_type == "text/html"

    def test_normal_field_without_alias(self) -> None:
        m = _ModelWithAlias.model_validate({"Content-Type": "a", "normal": "hello"})
        assert m.normal == "hello"

    def test_missing_attribute_raises(self) -> None:
        m = _ModelWithAlias.model_validate({"Content-Type": "a"})
        with pytest.raises(AttributeError, match="no_such_field"):
            _ = m.no_such_field

    def test_extra_field_resolves(self) -> None:
        """Extra fields (from extra='allow') are accessible normally."""
        m = _ModelWithAlias.model_validate(
            {"Content-Type": "a", "extra_key": "extra_val"}
        )
        assert m.extra_key == "extra_val"

    def test_serialize_by_alias(self) -> None:
        m = _ModelWithAlias.model_validate(
            {"Content-Type": "application/json", "X-Custom": 1}
        )
        dumped = m.model_dump()
        # serialize_by_alias=True means keys are the alias names.
        assert "Content-Type" in dumped
        assert "X-Custom" in dumped

    def test_model_config_extra_allow(self) -> None:
        assert _ModelWithAlias.model_config["extra"] == "allow"

    def test_model_config_serialize_by_alias(self) -> None:
        assert _ModelWithAlias.model_config["serialize_by_alias"] is True
