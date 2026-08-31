"""Tests for the QuickJS code interpreter and its PTC allowlist policy.

The allowlist is the whole security surface of this feature: the WASM guest has
no ambient capability, so anything the sandboxed JS reaches, it reached through
``ptc``. These cover what must be in it, what must stay out, and that the sandbox
boundary still holds for the file tools that are in it.

Requires the ``code-interpreter`` extra, which CI installs via
``uv sync --all-extras``.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Sequence

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool, tool

from uipath_langchain._utils.durable_interrupt import SUSPENDS_RUN
from uipath_langchain.agent.advanced import (
    PTC_FILESYSTEM_TOOLS,
    build_code_interpreter_middleware,
    create_advanced_agent,
    ptc_tool_names,
)

pytest.importorskip("langchain_quickjs", reason="needs the code-interpreter extra")


def _tool(name: str, *, suspends: bool = False) -> BaseTool:
    """A minimal agent tool, optionally flagged as suspending the run."""
    return StructuredTool.from_function(
        func=lambda value="": value,
        name=name,
        description=f"tool {name}",
        metadata={SUSPENDS_RUN: True} if suspends else {},
    )


class _ScriptedModel(GenericFakeChatModel):
    """Replays a fixed script and accepts any tool binding."""

    model_name: str = "test-model-code-interpreter"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        return {"ls_provider": "openai", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_ScriptedModel":
        return self


def _run_js(code: str, workspace: Path, tools: Sequence[BaseTool] = ()) -> str:
    """Run one ``eval`` call through a real advanced agent, return the tool output."""
    model = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "eval", "args": {"code": code}, "id": "c1"}],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    graph = create_advanced_agent(
        model=model,
        tools=list(tools),
        backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
        middleware=build_code_interpreter_middleware(list(tools)),
    )
    result = asyncio.run(
        graph.ainvoke({"messages": [{"role": "user", "content": "go"}]})
    )
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages, "the eval tool produced no output"
    return str(tool_messages[0].content)


# --------------------------------------------------------------------------
# Allowlist policy
# --------------------------------------------------------------------------


def test_suspending_tools_are_withheld() -> None:
    """A tool that suspends the run must never be reachable from the REPL.

    It cannot return a value into the JS ``await``, a replayed node re-runs every
    bridged call made before the interrupt, and PTC bypasses approval hooks.
    """
    assert ptc_tool_names(
        [_tool("read_invoice"), _tool("escalate", suspends=True)]
    ) == ["read_invoice"]


@pytest.mark.parametrize(
    "name",
    ["Get Invoice", "invoice.total", "2fa_check", "tool!", "faktura_\u010desk\u00e1"],
    ids=["space", "dot", "leading-digit", "punctuation", "non-ascii"],
)
def test_names_that_cannot_be_js_identifiers_are_withheld(name: str) -> None:
    """Dropped here rather than raising from inside ``wrap_model_call`` mid-run.

    Low-code tool names come from ``agent.json`` and are not constrained to
    JavaScript identifiers.
    """
    assert ptc_tool_names([_tool(name)]) == []


def test_camel_case_collisions_are_withheld() -> None:
    """Two tools that camel-case to one name are both dropped.

    Upstream dedupes by tool name, not camel name, so binding either would
    silently call the wrong tool.
    """
    assert ptc_tool_names([_tool("get_invoice"), _tool("get-invoice")]) == []


def test_reserved_task_name_is_withheld() -> None:
    """``task`` is the top-level ``task()`` global; listing it in ptc raises upstream."""
    assert ptc_tool_names([_tool("task")]) == []


def test_filesystem_tools_exposed_and_dangerous_ones_are_not() -> None:
    """Workspace file access is offered; ``delete``, ``execute`` and ``task`` are not.

    Asserted on the composed allowlist rather than the middleware's internals.
    That the allowlist actually reaches the sandbox is covered end to end by
    ``test_workspace_files_are_reachable_through_the_file_tools``.
    """
    exposed = {*ptc_tool_names([_tool("read_invoice")]), *PTC_FILESYSTEM_TOOLS}
    assert set(PTC_FILESYSTEM_TOOLS) <= exposed
    assert "read_invoice" in exposed
    assert exposed.isdisjoint({"delete", "execute", "task", "eval"})


def test_factory_returns_one_middleware() -> None:
    """The factory hands back exactly one entry, spliceable into a sequence."""
    assert len(build_code_interpreter_middleware([_tool("read_invoice")])) == 1


def test_factory_without_the_extra_raises_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extra must be genuinely optional.

    Importing ``uipath_langchain.agent.advanced`` has to keep working for every
    consumer that never asked for the code interpreter, so the middleware import
    is deferred into the factory. Setting the module to ``None`` in
    ``sys.modules`` is how the stdlib signals "absent", which is what a base
    install looks like.
    """
    monkeypatch.setitem(sys.modules, "langchain_quickjs", None)
    monkeypatch.setitem(sys.modules, "langchain_quickjs._ptc", None)

    with pytest.raises(ImportError, match="code-interpreter"):
        build_code_interpreter_middleware([_tool("read_invoice")])


def test_private_upstream_helpers_still_resolve() -> None:
    """Pins the private ``langchain_quickjs._ptc`` import the policy depends on.

    A local copy of the identifier rule risks drifting looser than upstream, and
    anything upstream rejects raises from inside ``wrap_model_call``. If this
    fails after a version bump, re-check the helpers before loosening the policy.
    """
    from langchain_quickjs._ptc import is_valid_ptc_tool_name, to_camel_case

    assert to_camel_case("read_file") == "readFile"
    assert is_valid_ptc_tool_name("read_file")
    assert not is_valid_ptc_tool_name("read file")


# --------------------------------------------------------------------------
# Sandbox behaviour, end to end through a real agent
# --------------------------------------------------------------------------


def test_computation_runs_in_the_sandbox(tmp_path: Path) -> None:
    """The plain arithmetic case: one eval call, no tools, a value back."""
    assert "<result>320</result>" in _run_js("10 * 32", tmp_path)


def test_programmatic_tool_calling_collapses_round_trips(tmp_path: Path) -> None:
    """Two bridged tool calls and the arithmetic between them, in one eval call."""
    seen: list[str] = []

    @tool
    def lookup_price(sku: str) -> str:
        """Look up the price of a SKU."""
        seen.append(sku)
        return {"A": "10", "B": "32"}[sku]

    code = """
    const [a, b] = await Promise.all([
      tools.lookupPrice({ sku: "A" }),
      tools.lookupPrice({ sku: "B" }),
    ]);
    Number(a) * Number(b)
    """
    assert "<result>320</result>" in _run_js(code, tmp_path, [lookup_price])
    assert sorted(seen) == ["A", "B"]


def test_workspace_files_are_reachable_through_the_file_tools(tmp_path: Path) -> None:
    """JS writes and reads a workspace file via the bridged file tools.

    This is what stands in for shell access: mediated by the tool, so the backend
    still resolves and bounds the path.
    """
    code = """
    await tools.writeFile({ file_path: "/note.txt", content: "hello" });
    await tools.readFile({ file_path: "/note.txt" })
    """
    assert "hello" in _run_js(code, tmp_path)
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "hello"


def test_path_traversal_is_still_rejected_through_the_bridge(tmp_path: Path) -> None:
    """``virtual_mode`` bounds the path even from inside the REPL.

    This is the property that makes bridged file tools an acceptable substitute
    for shell access: the backend, not the sandbox, resolves every path. The tool
    reports the refusal as a returned string, so the ``await`` resolves normally
    and the model sees the error.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    escape = tmp_path / "escaped.txt"
    code = """
    const r = await tools.writeFile({
      file_path: "/../escaped.txt", content: "pwned",
    });
    JSON.stringify(r)
    """
    output = _run_js(code, workspace)
    assert "Path traversal not allowed" in output
    assert not escape.exists(), f"traversal escaped the workspace: {escape}"
    assert list(workspace.rglob("*")) == [], "traversal wrote inside the workspace"


def test_sandbox_has_no_ambient_capability(tmp_path: Path) -> None:
    """No network, no module loader, no process: the guest starts with nothing."""
    code = "[typeof fetch, typeof require, typeof process].join(',')"
    assert "undefined,undefined,undefined" in _run_js(code, tmp_path)
