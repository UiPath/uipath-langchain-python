"""The REPL survives a process restart, so ``mode="thread"`` is honest.

Our advanced runs die at suspend: the graph checkpoints and a later resume is a
different process. If the QuickJS runtime lived only in the middleware instance,
every global and helper the model built would vanish at that boundary, silently,
and ``mode="turn"`` would be the truthful setting.

It does not. ``CodeInterpreterMiddleware`` declares a ``REPLState`` carrying
``_quickjs_slot_id`` plus an HMAC-signed snapshot payload on a ``DeltaChannel``,
all ``PrivateStateAttr``, so the checkpointer persists the interpreter's memory
and a resumed process replays it.

The subprocess test is the load-bearing one: two graphs in one interpreter would
pass even if the runtime were held in a process-level registry, so that variant
cannot tell persistence from a shared cache.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("langchain_quickjs", reason="needs the code-interpreter extra")
pytest.importorskip(
    "langgraph.checkpoint.sqlite.aio", reason="needs langgraph-checkpoint-sqlite"
)

# One turn in its own interpreter: build the agent, run one `eval`, print the
# result. Kept as source text rather than a helper module so the child shares
# nothing with the parent but the checkpoint file.
_TURN = """
import asyncio, sys
from typing import Any, Sequence
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from uipath_langchain.agent.advanced import build_code_interpreter_middleware

db, workspace, code = sys.argv[1], sys.argv[2], sys.argv[3]


class _Model(GenericFakeChatModel):
    model_name: str = "test-model-repl-persistence"

    def _get_ls_params(self, stop=None, **kwargs: Any) -> Any:
        return {"ls_provider": "openai", "ls_model_name": self.model_name}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_Model":
        return self


async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = create_deep_agent(
            model=_Model(messages=iter([
                AIMessage(content="", tool_calls=[
                    {"name": "eval", "args": {"code": code}, "id": "c"}
                ]),
                AIMessage(content="done"),
            ])),
            backend=FilesystemBackend(root_dir=workspace, virtual_mode=True),
            middleware=build_code_interpreter_middleware([]),
            checkpointer=saver,
        )
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "go"}]},
            {"configurable": {"thread_id": "t-1"}},
        )
        tool_messages = [m.content for m in result["messages"] if m.type == "tool"]
        print("RESULT:" + str(tool_messages[-1] if tool_messages else "none"))


asyncio.run(main())
"""


def _run_turn(script: Path, db: Path, workspace: Path, code: str) -> str:
    """Run one turn in a separate interpreter, return the eval output."""
    proc = subprocess.run(
        [sys.executable, str(script), str(db), str(workspace), code],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    assert proc.returncode == 0, f"turn failed:\n{proc.stderr[-2000:]}"
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")), None
    )
    assert line is not None, f"no RESULT line:\n{proc.stdout[-2000:]}"
    return line.removeprefix("RESULT:")


def test_repl_globals_survive_a_process_restart(tmp_path: Path) -> None:
    """A global set in one process is readable in the next, via the checkpoint."""
    script = tmp_path / "turn.py"
    script.write_text(textwrap.dedent(_TURN), encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    db = tmp_path / "state.db"

    first = _run_turn(script, db, workspace, "globalThis.marker = 42; 'set'")
    assert "set" in first, first

    second = _run_turn(
        script,
        db,
        workspace,
        "typeof globalThis.marker !== 'undefined'"
        " ? `SURVIVED ${globalThis.marker}` : 'LOST'",
    )
    assert "SURVIVED 42" in second, (
        f"REPL state did not cross the process boundary: {second!r}. If this is a"
        ' deliberate upstream change, mode="thread" is no longer honest and the'
        ' factory should pass mode="turn".'
    )


def test_snapshot_state_is_private_and_checkpointed() -> None:
    """Pins the state keys the persistence above depends on.

    If upstream renames or drops these, the subprocess test still catches the
    behaviour, but this says which contract broke.
    """
    from langchain_quickjs.middleware import REPLState

    annotations = REPLState.__annotations__
    assert "_quickjs_slot_id" in annotations
    assert "_quickjs_snapshot_payload" in annotations
    assert "_quickjs_snapshot_hmac" in annotations
