"""Tests for shared Pydantic model utilities."""

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, Field

from uipath_langchain._utils import get_unique_model_field_name


class _OccupiedNames(BaseModel):
    state_key: str
    state_key_1: str


def test_unique_model_field_name_uses_preferred_name_when_available() -> None:
    assert get_unique_model_field_name("state_key", None) == "state_key"


def test_unique_model_field_name_accepts_classes_and_instances() -> None:
    instance = _OccupiedNames(state_key="value", state_key_1="value")

    assert get_unique_model_field_name("state_key", instance) == "state_key_2"
    assert get_unique_model_field_name("state_key", _OccupiedNames) == "state_key_2"


@pytest.mark.parametrize(
    "validation_alias",
    [
        "state_key",
        AliasChoices("state_key", "value"),
        AliasPath("state_key", "value"),
    ],
)
def test_unique_model_field_name_avoids_validation_aliases(
    validation_alias: str | AliasChoices | AliasPath,
) -> None:
    class AliasedInput(BaseModel):
        value: str = Field(validation_alias=validation_alias)

    assert get_unique_model_field_name("state_key", AliasedInput) == "state_key_1"
