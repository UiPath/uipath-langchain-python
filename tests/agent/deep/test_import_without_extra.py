"""Verify a clear ImportError is raised when the [deep] extra is missing."""

import builtins
import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel


def _hide_deepagents(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "deepagents" or name.startswith("deepagents."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for mod in [
        m for m in sys.modules if m == "deepagents" or m.startswith("deepagents.")
    ]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_create_deep_agent_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_deepagents(monkeypatch)
    # Re-import the module so its lazy import path runs against the patched __import__.
    sys.modules.pop("uipath_langchain.agent.deep.agent", None)
    sys.modules.pop("uipath_langchain.agent.deep", None)
    deep_agent_module = importlib.import_module("uipath_langchain.agent.deep.agent")

    with pytest.raises(ImportError, match=r"uipath-langchain\[deep\]"):
        deep_agent_module.create_deep_agent(
            model=MagicMock(spec=BaseChatModel), system_prompt="x", tools=[]
        )


@pytest.mark.parametrize(
    "name", ["SubAgent", "CompiledSubAgent", "BackendProtocol", "BackendFactory"]
)
def test_lazy_reexports_raise_import_error(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    _hide_deepagents(monkeypatch)
    sys.modules.pop("uipath_langchain.agent.deep", None)
    deep_pkg = importlib.import_module("uipath_langchain.agent.deep")

    with pytest.raises(ImportError, match=r"uipath-langchain\[deep\]"):
        getattr(deep_pkg, name)
