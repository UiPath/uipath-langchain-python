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

_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _decorator_names(fn: _Function) -> set[str]:
    names = set()
    for decorator in fn.decorator_list:
        node = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _suspends(fn: _Function) -> bool:
    """Whether ``fn`` or anything nested in it interrupts the run.

    Both shapes count: the ``durable_interrupt`` decorator, and a bare
    ``interrupt()`` call, which ``create_ixp_extraction_tool`` uses.
    """
    for node in ast.walk(fn):
        if isinstance(node, _Function) and "durable_interrupt" in _decorator_names(
            node
        ):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "interrupt"
        ):
            return True
    return False


def _stamps(fn: _Function) -> bool:
    """Whether ``fn`` sets ``SUSPENDS_RUN`` as a dict key to a true constant.

    Parsed rather than grepped so a mention in a comment or docstring does not
    count as a stamp.
    """
    for node in ast.walk(fn):
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


def _suspending_factories() -> list[tuple[str, _Function]]:
    """Every top-level factory under the tools package that suspends the run.

    Scoped per factory, not per module: ``context_tool`` holds two suspending
    builders next to a non-suspending one, so a module-wide answer would let a
    third suspending builder pass on a sibling's stamp.
    """
    found = []
    for path in sorted(_TOOLS_DIR.rglob("*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for fn in module.body:
            if isinstance(fn, _Function) and _suspends(fn):
                found.append((f"{path.name}::{fn.name}", fn))
    assert found, f"no suspending tool factories found under {_TOOLS_DIR}"
    return found


@pytest.mark.parametrize(
    ("factory", "node"),
    _suspending_factories(),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_suspending_factory_stamps_the_flag(factory: str, node: _Function) -> None:
    """A factory that suspends the run stamps ``SUSPENDS_RUN: True`` in metadata."""
    assert _stamps(node), (
        f"{factory} suspends the run but does not set {SUSPENDS_RUN!r} in its "
        f"tool metadata. Add `SUSPENDS_RUN: True` to the tool's metadata dict, "
        f"or the tool becomes callable from the code interpreter's tools namespace."
    )
