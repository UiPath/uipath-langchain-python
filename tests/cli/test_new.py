import os
import re
from importlib.metadata import version

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from uipath._cli.models.agent_frameworks import AgentFramework
from uipath._cli.models.project_types import ProjectType

from uipath_langchain._cli.cli_new import (
    UIPATH_LANGCHAIN_SCAFFOLD_MINOR,
    langgraph_new_middleware,
)

PIN_RE = re.compile(r'"uipath-langchain\[bedrock,vertex\]([^"]*)"')


class TestProjectTypeGate:
    """The middleware only claims agent scaffolds; everything else passes through."""

    def test_function_type_passes_through(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """project_type='function' must defer to the base `uipath new` scaffold.

        Regression guard for uipath-python#1543: installing uipath-langchain
        used to hijack `uipath new` unconditionally, making the base function
        scaffold unreachable.
        """
        monkeypatch.chdir(tmp_path)
        result = langgraph_new_middleware("demo", project_type=ProjectType.FUNCTION)
        assert result.should_continue is True
        assert not os.path.exists("main.py")
        assert not os.path.exists("langgraph.json")
        assert not os.path.exists("pyproject.toml")

    def test_auto_type_scaffolds_agent(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--type auto (the default): an installed framework claims the scaffold."""
        monkeypatch.chdir(tmp_path)
        result = langgraph_new_middleware("demo", project_type=ProjectType.AUTO)
        assert result.should_continue is False
        assert os.path.exists("main.py")
        assert os.path.exists("langgraph.json")

    def test_auto_type_plain_string_scaffolds_agent(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raw-string 'auto' from other callers is claimed the same way."""
        monkeypatch.chdir(tmp_path)
        # Deliberately off-type: the point is that a plain string still gates.
        result = langgraph_new_middleware("demo", project_type="auto")  # type: ignore[arg-type]
        assert result.should_continue is False
        assert os.path.exists("langgraph.json")

    def test_agent_type_scaffolds_agent(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = langgraph_new_middleware(
            "demo",
            project_type=ProjectType.AGENT,
            agent_framework=AgentFramework.LANGCHAIN,
        )
        assert result.should_continue is False
        assert os.path.exists("main.py")
        assert os.path.exists("langgraph.json")

    def test_other_agent_framework_passes_through(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another framework's scaffold must be left to its own integration."""
        monkeypatch.chdir(tmp_path)
        result = langgraph_new_middleware(
            "demo",
            project_type=ProjectType.AGENT,
            agent_framework=AgentFramework.PYDANTIC_AI,
        )
        assert result.should_continue is True
        assert not os.path.exists("main.py")
        assert not os.path.exists("langgraph.json")
        assert not os.path.exists("pyproject.toml")

    def test_plain_string_arguments_still_gate_correctly(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers passing raw strings (older CLIs) must hit the same gate."""
        monkeypatch.chdir(tmp_path)
        # Deliberately off-type: the point is that plain strings still gate.
        result = langgraph_new_middleware(
            "demo",
            project_type="agent",  # type: ignore[arg-type]
            agent_framework="pydantic-ai",  # type: ignore[arg-type]
        )
        assert result.should_continue is True
        assert not os.path.exists("langgraph.json")

    def test_default_type_stays_agent_for_old_base_cli(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Older `uipath` versions call without project_type; keep scaffolding an agent."""
        monkeypatch.chdir(tmp_path)
        result = langgraph_new_middleware("demo")
        assert result.should_continue is False
        assert os.path.exists("langgraph.json")


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
