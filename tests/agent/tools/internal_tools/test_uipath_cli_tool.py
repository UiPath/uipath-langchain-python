"""Tests for the sandboxed ``uipath_cli`` built-in tool."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from uipath.runtime import Workspace

from uipath_langchain.agent.tools.internal_tools.uipath_cli_tool import (
    create_uipath_cli_tool,
)


def _workspace(path: Path) -> Workspace:
    return Workspace(path, cleanup=False)


def _invoke(tool: Any, **kwargs: Any) -> str:
    return tool.invoke(kwargs)


class TestCreateUipathCliTool:
    def test_does_not_create_workspace(self, tmp_path: Path) -> None:
        # The runtime owns workspace creation; the tool must not create directories.
        target = tmp_path / "does" / "not" / "exist"
        create_uipath_cli_tool(_workspace(target))
        assert not target.exists()

    def test_marks_tool_as_internal(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "internal"
        assert tool.metadata["display_name"] == "uipath_cli"

    def test_rejects_disallowed_binary(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        result = _invoke(tool, command="rm -rf /")
        assert "not allowed" in result
        assert "rm" in result

    def test_rejects_path_qualified_binary(self, tmp_path: Path) -> None:
        # A lookalike 'uip' inside the workspace must not be runnable via a path.
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        for command in ("./uip login", "/tmp/uip login", "bin/uipath pack"):
            result = _invoke(tool, command=command)
            assert "not allowed" in result, command

    def test_rejects_subdir_escaping_workspace(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        result = _invoke(tool, command="uipath pack", subdir="../outside")
        assert "escapes the workspace" in result

    def test_reports_missing_binary(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        with patch(
            "uipath_langchain.agent.tools.internal_tools.uipath_cli_tool.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            result = _invoke(tool, command="uipath --version")
        assert "not found on PATH" in result

    def test_rejects_denied_subcommand_by_default(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        result = _invoke(tool, command="uip solution delete MySolution")
        assert "not allowed" in result
        assert "delete" in result

    def test_rejects_hyphenated_destructive_variant(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        result = _invoke(tool, command="uip or queue-items delete-bulk")
        assert "not allowed" in result
        assert "delete-bulk" in result

    def test_custom_denied_subcommands_override_default(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(
            _workspace(tmp_path), denied_subcommands=["publish"]
        )
        # 'delete' is no longer denied once the default set is overridden.
        with patch(
            "uipath_langchain.agent.tools.internal_tools.uipath_cli_tool.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            allowed = _invoke(tool, command="uip delete asset X")
        assert "not found on PATH" in allowed
        blocked = _invoke(tool, command="uip solution publish dev")
        assert "not allowed" in blocked
        assert "publish" in blocked

    def test_does_not_block_benign_commands(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path))
        with patch(
            "uipath_langchain.agent.tools.internal_tools.uipath_cli_tool.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            for command in (
                "uip solution deploy dev",
                "uip solution publish dev",
                "uipath pack",
                "uip login",
            ):
                result = _invoke(tool, command=command)
                assert "not allowed" not in result, command

    def test_custom_timeout_is_used(self, tmp_path: Path) -> None:
        tool = create_uipath_cli_tool(_workspace(tmp_path), timeout_sec=5)
        with patch(
            "uipath_langchain.agent.tools.internal_tools.uipath_cli_tool.subprocess.run",
        ) as mock_run:
            mock_run.return_value = type(
                "P", (), {"returncode": 0, "stdout": "", "stderr": ""}
            )()
            _invoke(tool, command="uipath --version")
        assert mock_run.call_args.kwargs["timeout"] == 5
