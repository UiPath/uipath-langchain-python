from types import SimpleNamespace
from typing import Any

import pytest

from uipath_langchain.chat.thinking import model_rejects_forced_tool_choice


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"shouldSkipForcedToolChoice": True}, True),
        ({"shouldSkipForcedToolChoice": False}, False),
        ({}, False),
        (None, False),
    ],
)
def test_model_rejects_forced_tool_choice_reads_discovery_flag(
    details: Any, expected: bool
) -> None:
    assert (
        model_rejects_forced_tool_choice(SimpleNamespace(model_details=details))
        is expected
    )


def test_model_without_model_details_does_not_reject() -> None:
    assert model_rejects_forced_tool_choice(object()) is False
