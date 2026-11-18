"""The lock keeps the runtime the QuickJS shutdown fix depends on.

langchain-quickjs 0.3.12 shuts its REPL registry down on its own, which is what
lets the package drop its exit workaround, and it needs langchain-core 1.7.5.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[1]


def _locked_versions(name: str) -> list[Version]:
    with (_ROOT / "uv.lock").open("rb") as f:
        packages = tomllib.load(f)["package"]
    return [Version(p["version"]) for p in packages if p["name"] == name]


@pytest.mark.parametrize(
    ("name", "floor"),
    [("langchain-core", "1.7.5"), ("langchain-quickjs", "0.3.12")],
)
def test_locked_runtime_meets_the_shutdown_fix_floor(name: str, floor: str) -> None:
    versions = _locked_versions(name)

    assert versions
    assert min(versions) >= Version(floor)
