"""Contract test: subagents receive the parent's tools, create-file included.

Deliberately **not** mocked. Every other test in this directory patches
``_create_deep_agent``, so they assert what we pass in and never what deepagents
does with it. Only a real graph catches an upstream change to how subagent tool
lists are resolved.

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
from uipath.agent.models.agent import AgentInternalToolResourceConfig

from uipath_langchain.agent.advanced.agent import create_advanced_agent
from uipath_langchain.agent.tools.internal_tools.create_file_tool import (
    create_file_tool,
)

CREATE_FILE_TOOL_NAME = "Make_Report"

_BINDINGS: list[list[str]] = []


def _create_file_tool() -> BaseTool:
    return create_file_tool(
        AgentInternalToolResourceConfig.model_validate(
            {
                "$resourceType": "tool",
                "type": "Internal",
                "name": "Make Report",
                "description": "Make a file.",
                "properties": {"toolType": "create-file"},
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fileName": {"type": "string"},
                        "content": {"type": "string"},
                        "filePath": {"type": "string"},
                    },
                    "required": ["fileName"],
                },
            }
        )
    )


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
        tools=[_create_file_tool(), _tool("read_invoice")],
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
def test_every_agent_holds_the_create_file_tool(
    tmp_path: Path, subagent_type: str, subagents: Sequence[SubAgent]
) -> None:
    """``general-purpose`` is covered too: we replace deepagents' implicit spec."""
    bindings = _dispatch(tmp_path, subagent_type, subagents)
    main = [b for b in bindings if "task" in b]
    subagent = [b for b in bindings if "task" not in b]

    assert main, f"no main-agent binding found: {bindings}"
    assert subagent, f"no subagent binding found: {bindings}"
    assert all(CREATE_FILE_TOOL_NAME in b for b in main), main
    assert all(CREATE_FILE_TOOL_NAME in b for b in subagent), subagent


@pytest.mark.parametrize(
    ("subagent_type", "subagents"),
    [("worker", (_WORKER,)), (GENERAL_PURPOSE_SUBAGENT["name"], ())],
    ids=["declared-subagent", "general-purpose"],
)
def test_other_tools_reach_every_agent(
    tmp_path: Path, subagent_type: str, subagents: Sequence[SubAgent]
) -> None:
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


def test_a_subagent_declaring_its_own_tools_still_gets_our_middleware() -> None:
    from uipath_langchain.agent.advanced.agent import _resolve_subagent_specs
    from uipath_langchain.agent.advanced.job_attachments_middleware import (
        JobAttachmentsMiddleware,
    )

    middleware = JobAttachmentsMiddleware()
    (spec, _general_purpose) = _resolve_subagent_specs(
        [{**_WORKER, "tools": [_tool("only_mine")]}], [], None, [middleware]
    )

    resolved: dict[str, Any] = dict(spec)
    assert [getattr(t, "name", None) for t in resolved["tools"]] == ["only_mine"]
    assert middleware in resolved["middleware"]
