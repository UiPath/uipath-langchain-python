import os
import re
import tomllib
from importlib.metadata import version
from pathlib import Path

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


TEMPLATE_DIR = Path(__file__).parents[2] / "template"


class TestTemplatePin:
    """``template/`` must track the same minor as ``uipath new``.

    Studio Web builds its coded agent sample from ``template/``, so a stale pin
    there ships an old uipath-langchain to new projects. When these fail,
    update the pin in ``template/pyproject.toml`` and run ``uv lock`` in
    ``template/``.
    """

    def test_template_pin_admits_installed_version(self) -> None:
        content = (TEMPLATE_DIR / "pyproject.toml").read_text()
        match = PIN_RE.search(content)
        assert match is not None, content
        installed = version("uipath-langchain")
        assert SpecifierSet(match.group(1)).contains(installed, prereleases=True), (
            f"template/pyproject.toml pin '{match.group(1)}' does not contain "
            f"uipath-langchain {installed}"
        )

    def test_template_lock_is_on_the_scaffold_minor(self) -> None:
        lock = tomllib.loads((TEMPLATE_DIR / "uv.lock").read_text())
        [locked] = [
            Version(package["version"])
            for package in lock["package"]
            if package["name"] == "uipath-langchain"
        ]
        assert f"{locked.major}.{locked.minor}" == UIPATH_LANGCHAIN_SCAFFOLD_MINOR, (
            f"template/uv.lock locks uipath-langchain {locked}, not "
            f"{UIPATH_LANGCHAIN_SCAFFOLD_MINOR}.x"
        )
