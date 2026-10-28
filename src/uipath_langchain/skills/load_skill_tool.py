"""The ``load_skill`` tool: a coded agent's on-demand access to a bound skill's instructions."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model
from uipath.platform.skills import Skill, SkillBundle, SkillSet

_SKILL_MD = "SKILL.md"
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_BINARY_SNIFF_CHARS = 8 * 1024

MAX_OUTPUT_CHARS = 64_000
"""Longest file text the tool returns; longer text is cut and marked as truncated."""

MAX_ENUM_NAMES = 200
"""Most skill names listed as an enum in the tool schema; above it ``name`` is a plain string."""


def _render(bundle: SkillBundle, skill: Skill, path: str, text: str) -> str:
    """The tool output for one file: a header the model can cite, then the text."""
    qualified = f"{bundle.ref.name}/{skill.name}"
    if path == _SKILL_MD:
        others = [f for f in skill.files if f != _SKILL_MD]
        footer = f"\n\nSupporting files: {', '.join(others)}" if others else ""
        return f"# {qualified} (bundle {bundle.ref.name} v{bundle.version})\n\n{text}{footer}"
    return f"# {qualified} — {path}\n\n{text}"


def _validate_tool_name(tool_name: str) -> None:
    """Raise ``ValueError`` unless ``tool_name`` is 1-64 letters, digits, ``_`` or ``-``."""
    if not _TOOL_NAME.fullmatch(tool_name):
        raise ValueError(
            f"tool_name must be 1-64 letters, digits, '_' or '-': {tool_name!r}"
        )


def _truncate(text: str) -> str:
    """``text`` cut to ``MAX_OUTPUT_CHARS``, with a note saying how much was left out."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        f"{text[:MAX_OUTPUT_CHARS]}\n\n"
        f"[truncated: showing {MAX_OUTPUT_CHARS} of {len(text)} characters]"
    )


def create_load_skill_tool(
    skill_set: SkillSet, *, tool_name: str = "load_skill"
) -> StructuredTool:
    """A tool that returns a skill's ``SKILL.md`` (or one of its files) by qualified name.

    The model-facing ``name`` argument is an enum of the set's qualified
    ``<bundle>/<skill>`` names, so the model cannot ask for a skill that is not
    bound; with no skills, or more than ``MAX_ENUM_NAMES``, it is a plain
    string. Errors are returned as text so the model can recover; nothing is
    raised past the tool node. Text longer than ``MAX_OUTPUT_CHARS`` is
    truncated and non-text files are refused. Reading a ``SKILL.md`` records
    the SDK's ``Skill load`` span.

    Args:
        skill_set: The skills bound to the agent, materialized to disk.
        tool_name: The tool name the model sees; also its ``display_name``.

    Returns:
        A ``StructuredTool`` with sync and async implementations and
        ``metadata={"tool_type": "internal", "display_name": tool_name}``.

    Raises:
        ValueError: ``tool_name`` is not 1-64 letters, digits, ``_`` or ``-``.
    """
    _validate_tool_name(tool_name)
    names = skill_set.names()
    name_type: Any = (
        Literal[tuple(names)] if names and len(names) <= MAX_ENUM_NAMES else str
    )
    args = create_model(
        "LoadSkillInput",
        name=(
            name_type,
            Field(
                description="The skill to load, as `<bundle>/<skill>` from the Skills list."
            ),
        ),
        path=(
            str,
            Field(
                default=_SKILL_MD,
                description="A file inside the skill folder. Default: its SKILL.md.",
            ),
        ),
    )

    def load(name: str, path: str = _SKILL_MD) -> str:
        hit = skill_set.find(name)
        if hit is None:
            return f"Unknown skill '{name}'. Available: {', '.join(names) or 'none'}"
        bundle, skill = hit
        path = path.replace("\\", "/").removeprefix("./")
        try:
            text = skill_set.read(name, path)
        except KeyError:
            return (
                f"Skill '{name}' has no file '{path}'. Files: {', '.join(skill.files)}"
            )
        except RuntimeError as e:
            return f"Skill '{name}' is not available any more: {e}"
        except OSError as e:
            # strerror only: the exception's own text carries the local file path.
            return f"Skill '{name}' could not be read: {e.strerror or type(e).__name__}"
        if "\x00" in text[:_BINARY_SNIFF_CHARS]:
            return (
                f"Skill '{name}' file '{path}' is not a text file and cannot be shown."
            )
        return _render(bundle, skill, path, _truncate(text))

    async def aload(name: str, path: str = _SKILL_MD) -> str:
        return await asyncio.to_thread(load, name, path)

    return StructuredTool.from_function(
        func=load,
        coroutine=aload,
        name=tool_name,
        args_schema=args,
        description=(
            "Load the full instructions of one of the bound skills before "
            "following it. `name` is `<bundle>/<skill>`."
        ),
        metadata={"tool_type": "internal", "display_name": tool_name},
    )
