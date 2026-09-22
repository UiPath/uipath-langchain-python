"""Tests for generic tool dispatch."""

from unittest.mock import MagicMock

import pytest
from uipath.agent.models.agent import (
    AgentGenericToolProperties,
    AgentGenericToolResourceConfig,
)

from uipath_langchain.agent.exceptions import AgentStartupError
from uipath_langchain.agent.tools.generic_tools import create_generic_tool
from uipath_langchain.agent.tools.tool_factory import _build_tool_for_resource

JEV_SETTINGS = {
    "questions": [
        {
            "name": "is_urgent",
            "type": "noul",
            "instructions": "The message conveys urgency",
        }
    ]
}


def _resource(sub_type: str) -> AgentGenericToolResourceConfig:
    return AgentGenericToolResourceConfig(
        name="Classify ticket",
        description="Classify a ticket",
        input_schema={},
        properties=AgentGenericToolProperties(sub_type=sub_type, settings=JEV_SETTINGS),
    )


@pytest.fixture(autouse=True)
def _enable_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UIPATH_FEATURE_EnableJevTool", "true")


@pytest.mark.parametrize("sub_type", ["jev", "JEV"])
def test_dispatches_jev_subtype(sub_type: str) -> None:
    tool = create_generic_tool(_resource(sub_type), MagicMock())
    assert tool.name == "Classify_ticket"
    assert tool.metadata is not None
    assert tool.metadata["tool_type"] == "generic"
    assert tool.metadata["sub_type"] == "jev"


def test_unknown_subtype_raises_startup_error() -> None:
    with pytest.raises(AgentStartupError) as exc_info:
        create_generic_tool(_resource("not-a-real-subtype"), MagicMock())
    assert "not-a-real-subtype" in str(exc_info.value.error_info.detail)


async def test_tool_factory_routes_generic_resources() -> None:
    tool = await _build_tool_for_resource(_resource("jev"), MagicMock())
    assert tool is not None
    assert not isinstance(tool, list)
    assert tool.name == "Classify_ticket"
