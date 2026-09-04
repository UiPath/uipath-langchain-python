import os
import re
from importlib.metadata import version

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from uipath_langchain._cli.cli_new import (
    UIPATH_LANGCHAIN_SCAFFOLD_MINOR,
    langgraph_new_middleware,
)

PIN_RE = re.compile(r'"uipath-langchain\[bedrock,vertex\]([^"]*)"')


class TestUipathLangchainScaffoldPin:
    """The scaffolded pin must admit the uipath-langchain release it ships with."""

    def test_scaffold_pin_admits_installed_version(self) -> None:
        """Guard: fails on every minor bump so the scaffold gets reviewed.

        When this fails, review the scaffold in ``cli_new.py`` (pin constant,
        ``main.py`` template, post-scaffold hints) for the new minor, then bump
        ``UIPATH_LANGCHAIN_SCAFFOLD_MINOR``.
        """
        installed = Version(version("uipath-langchain"))
        installed_minor = f"{installed.major}.{installed.minor}"
        assert UIPATH_LANGCHAIN_SCAFFOLD_MINOR == installed_minor, (
            f"uipath-langchain minor changed to {installed_minor} but "
            f"UIPATH_LANGCHAIN_SCAFFOLD_MINOR is {UIPATH_LANGCHAIN_SCAFFOLD_MINOR}; "
            f"review the scaffold in cli_new.py (pin, template, hints) and bump "
            f"the constant"
        )

    def test_scaffolded_pin_contains_installed_version(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: runs against the real installed package, not a mock.

        A stale range would make ``uv sync`` downgrade the project's venv right
        after ``uipath new``.
        """
        monkeypatch.chdir(tmp_path)
        result = langgraph_new_middleware("demo")
        assert result.should_continue is False
        assert os.path.exists("main.py")
        assert os.path.exists("langgraph.json")
        content = (tmp_path / "pyproject.toml").read_text()
        match = PIN_RE.search(content)
        assert match is not None, content
        installed = version("uipath-langchain")
        assert SpecifierSet(match.group(1)).contains(installed, prereleases=True), (
            f"scaffolded pin '{match.group(1)}' does not contain installed "
            f"uipath-langchain {installed}"
        )
