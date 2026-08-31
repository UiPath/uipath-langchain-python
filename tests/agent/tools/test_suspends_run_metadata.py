"""Every tool factory that suspends the run must advertise it in tool metadata.

A suspending tool raises ``GraphInterrupt`` instead of returning: the run
checkpoints and the node is replayed from that checkpoint on resume. Callers that
invoke tools outside the graph's tool node -- the QuickJS code interpreter's
programmatic tool calling in particular -- must therefore not offer them, because
a replayed node re-runs every call made before the interrupt, and because such
bridges bypass approval hooks.

Deciding eligibility from ``SUSPENDS_RUN`` keeps that policy next to the code that
suspends, rather than in a central list that silently goes stale. This test is
what makes the flag trustworthy: it reads the factory sources, so a new
suspending factory that forgets to stamp it fails here instead of quietly
becoming reachable from inside the sandbox.
"""

import ast
from pathlib import Path

import pytest

from uipath_langchain._utils.durable_interrupt import SUSPENDS_RUN

_TOOLS_DIR = Path(__file__).parents[3] / "src" / "uipath_langchain" / "agent" / "tools"

# Suspends the run without the decorator: a bare ``interrupt()`` call.
_BARE_INTERRUPT_MODULES = {"extraction_tool.py"}


def _suspending_modules() -> list[Path]:
    """Factory modules that suspend the run, found by reading the source."""
    found = [
        path
        for path in sorted(_TOOLS_DIR.rglob("*.py"))
        if "@durable_interrupt" in path.read_text(encoding="utf-8")
        or path.name in _BARE_INTERRUPT_MODULES
    ]
    assert found, f"no suspending tool factories found under {_TOOLS_DIR}"
    return found


def _assigns_suspends_run(source: str) -> bool:
    """Whether the module sets ``SUSPENDS_RUN`` as a dict key to a true constant.

    Parsed rather than grepped so a mention in a comment or docstring does not
    count as a stamp.
    """
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if (
                isinstance(key, ast.Name)
                and key.id == "SUSPENDS_RUN"
                and isinstance(value, ast.Constant)
                and value.value is True
            ):
                return True
    return False


@pytest.mark.parametrize("module", _suspending_modules(), ids=lambda p: p.name)
def test_suspending_factory_stamps_the_flag(module: Path) -> None:
    """A factory that suspends the run stamps ``SUSPENDS_RUN: True`` in metadata."""
    assert _assigns_suspends_run(module.read_text(encoding="utf-8")), (
        f"{module.name} suspends the run but does not set "
        f"{SUSPENDS_RUN!r} in its tool metadata. Add "
        f"`SUSPENDS_RUN: True` to the tool's metadata dict, or the tool becomes "
        f"callable from the code interpreter's tools namespace."
    )
