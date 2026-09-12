"""Contract test: main-agent-only tools must never reach a subagent.

Deliberately **not** mocked. Every other test in this directory patches
``_create_deep_agent``, so they assert what we pass in and never what deepagents
does with it. Only a real graph catches an upstream change that starts sharing the
parent tool list with subagents again.

The bug this guards: a subagent that calls ``create_output_file`` uploads a real job
attachment and returns prose. The reference never reaches the main agent, the only
agent that fills the typed output, so the main agent uploads a second orphan
attachment and the job faults.

Bindings are recorded per ``bind_tools`` call rather than per model, because a
subagent with no ``model`` in its spec inherits the parent's instance -- so the
main agent and the general-purpose subagent are the same object. The main agent is
told apart by holding ``task``: only an agent that can dispatch subagents gets it,
and it binds once per turn.
"""

import asyncio
from pathlib import Path
from typing import Any, Sequence

import pytest
from deepagents import SubAgent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool

from uipath_langchain.agent.advanced.agent import (
    MAIN_AGENT_ONLY_TOOLS,
    create_advanced_agent,
)
from uipath_langchain.agent.attachments.constants import OUTPUT_FILE_TOOL_NAME

_BINDINGS: list[list[str]] = []


def _tool(name: str) -> BaseTool:
    return StructuredTool.from_function(
        func=lambda value="": value, name=name, description=f"tool {name}"
    )


class _RecordingModel(GenericFakeChatModel):
    """Appends the tool names of every bind_tools call to a module-level sink."""

    model_name: str = "test-model-main-only"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return {"ls_provider": "openai", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_RecordingModel":
        _BINDINGS.append(sorted(t.name for t in tools))
        return self


def _dispatch(
    tmp_path: Path,
    subagent_type: str,
    subagents: Sequence[SubAgent] = (),
) -> list[list[str]]:
    """Build a real deep agent, dispatch to ``subagent_type``, return all bindings."""
    _BINDINGS.clear()
    model = _RecordingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "go",
                                "subagent_type": subagent_type,
                            },
                            "id": "c1",
                        }
                    ],
                ),
                *[AIMessage(content="done")] * 20,
            ]
        )
    )
    graph = create_advanced_agent(
        model=model,
        tools=[_tool(OUTPUT_FILE_TOOL_NAME), _tool("read_invoice")],
        subagents=[SubAgent(**{**s, "model": model}) for s in subagents],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
    )
    asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "hi"}]}))
    assert len(_BINDINGS) >= 2, (
        f"expected a main and a subagent binding, got {_BINDINGS}"
    )
    return list(_BINDINGS)


_WORKER: SubAgent = {
    "name": "worker",
    "description": "does work",
    "system_prompt": "work",
}


@pytest.mark.parametrize(
    ("subagent_type", "subagents"),
    [("worker", (_WORKER,)), (GENERAL_PURPOSE_SUBAGENT["name"], ())],
    ids=["declared-subagent", "general-purpose"],
)
def test_only_the_main_agent_holds_the_output_file_tool(
    tmp_path: Path, subagent_type: str, subagents: Sequence[SubAgent]
) -> None:
    """The main agent holds it, the dispatched subagent does not.

    ``general-purpose`` is the load-bearing case: deepagents adds it implicitly and
    would otherwise hand it the parent tool list.
    """
    bindings = _dispatch(tmp_path, subagent_type, subagents)
    main = [b for b in bindings if "task" in b]
    subagent = [b for b in bindings if "task" not in b]

    assert main, f"no main-agent binding found: {bindings}"
    assert subagent, f"no subagent binding found: {bindings}"
    assert all(OUTPUT_FILE_TOOL_NAME in b for b in main), main
    assert not any(OUTPUT_FILE_TOOL_NAME in b for b in subagent), subagent


@pytest.mark.parametrize(
    ("subagent_type", "subagents"),
    [("worker", (_WORKER,)), (GENERAL_PURPOSE_SUBAGENT["name"], ())],
    ids=["declared-subagent", "general-purpose"],
)
def test_shared_tools_still_reach_every_agent(
    tmp_path: Path, subagent_type: str, subagents: Sequence[SubAgent]
) -> None:
    """Withholding one tool must not withhold the rest."""
    bindings = _dispatch(tmp_path, subagent_type, subagents)
    assert all("read_invoice" in b for b in bindings), bindings


def test_a_subagent_declaring_its_own_tools_is_left_alone(tmp_path: Path) -> None:
    """An explicit ``tools`` on a spec is the caller's decision, not ours to rewrite.

    Its list replaces the parent's rather than merging with it, so the subagent sees
    ``only_mine`` and not the parent's ``read_invoice``. The filesystem tools are
    still present because ``FilesystemMiddleware`` adds those to every agent.
    """
    bindings = _dispatch(
        tmp_path, "worker", ({**_WORKER, "tools": [_tool("only_mine")]},)
    )
    subagent = [b for b in bindings if "task" not in b]
    assert subagent, f"no subagent binding found: {bindings}"
    assert all("only_mine" in b for b in subagent), subagent
    assert not any("read_invoice" in b for b in subagent), subagent


def test_the_withheld_set_is_not_empty() -> None:
    """Guard against the set being emptied and the tests above passing vacuously."""
    assert OUTPUT_FILE_TOOL_NAME in MAIN_AGENT_ONLY_TOOLS
