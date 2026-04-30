"""Smoke test for create_deep_agent.

Verifies create_deep_agent forwards its arguments to deepagents.create_deep_agent.
We don't exercise the deepagents internals (those are tested by the deepagents
package itself); we only validate UiPath's pass-through.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("deepagents")

from uipath_langchain.agent.deep import create_deep_agent  # noqa: E402


def test_create_deep_agent_forwards_to_deepagents() -> None:
    sentinel_graph = MagicMock(name="compiled_deep_agent")
    fake_upstream = MagicMock(return_value=sentinel_graph)
    model = MagicMock()

    with patch(
        "uipath_langchain.agent.deep.agent._import_create_deep_agent",
        return_value=fake_upstream,
    ):
        graph = create_deep_agent(
            model=model, system_prompt="sys", tools=[], subagents=[]
        )

    assert graph is sentinel_graph
    fake_upstream.assert_called_once()
    kwargs = fake_upstream.call_args.kwargs
    assert kwargs["model"] is model
    assert kwargs["system_prompt"] == "sys"
    assert kwargs["tools"] == []
    assert kwargs["subagents"] == []
    assert kwargs["backend"] is None
    assert kwargs["response_format"] is None
