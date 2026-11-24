"""The ``load_skill`` tool: a coded agent's on-demand access to a bound skill's instructions."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.tools import StructuredTool, ToolException
from pydantic import Field, create_model
from uipath.platform.skills import Skill, SkillBundle, SkillSet

_SKILL_MD = "SKILL.md"
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_BINARY_SNIFF_CHARS = 8 * 1024

MAX_OUTPUT_CHARS = 64_000
"""Longest response the tool returns; a longer file is cut and marked as truncated."""

MAX_LIST_CHARS = 2_000
"""Longest list of names (supporting files, skills) a response shows; the rest are counted."""

MAX_ENUM_NAMES = 200
"""Most skill names listed as an enum in the tool schema; above it ``name`` is a plain string."""

_TRUNCATION_NOTE_CHARS = 64
"""Room kept for the truncation note, so a cut response stays within ``MAX_OUTPUT_CHARS``."""


def _render(bundle: SkillBundle, skill: Skill, path: str, text: str) -> str:
    """The tool output for one file: a header the model can cite, then the text.

    A ``SKILL.md`` ends with the list of the skill's other files. The text is
    cut so that the whole output stays within ``MAX_OUTPUT_CHARS``.
    """
    qualified = f"{bundle.ref.name}/{skill.name}"
    footer = ""
    if path == _SKILL_MD:
        header = f"# {qualified} (bundle {bundle.ref.name} v{bundle.version})\n\n"
        others = [f for f in skill.files if f != _SKILL_MD]
        if others:
            footer = f"\n\nSupporting files: {_name_list(others)}"
    else:
        header = f"# {qualified} — {path}\n\n"
    room = MAX_OUTPUT_CHARS - len(header) - len(footer)
    return header + _truncate(text, room) + footer


def _name_list(names: Sequence[str]) -> str:
    """``names`` joined by commas, up to ``MAX_LIST_CHARS``; the names left out are counted."""
    shown: list[str] = []
    length = 0
    for name in names:
        length += len(name) + (2 if shown else 0)
        if length > MAX_LIST_CHARS:
            break
        shown.append(name)
    rest = len(names) - len(shown)
    if not rest:
        return ", ".join(shown)
    if not shown:
        return f"{rest} (names too long to list)"
    return f"{', '.join(shown)}, and {rest} more"


def _validate_tool_name(tool_name: str) -> None:
    """Raise ``ValueError`` unless ``tool_name`` is 1-64 letters, digits, ``_`` or ``-``."""
    if not _TOOL_NAME.fullmatch(tool_name):
        raise ValueError(
            f"tool_name must be 1-64 letters, digits, '_' or '-': {tool_name!r}"
        )


def _truncate(text: str, limit: int) -> str:
    """``text`` cut to fit in ``limit`` characters, with a note saying how much was left out."""
    if len(text) <= limit:
        return text
    shown = max(limit - _TRUNCATION_NOTE_CHARS, 0)
    return f"{text[:shown]}\n\n[truncated: showing {shown} of {len(text)} characters]"


def create_load_skill_tool(
    skill_set: SkillSet, *, tool_name: str = "load_skill"
) -> StructuredTool:
    """A tool that returns a skill's ``SKILL.md`` (or one of its files) by qualified name.

    The model-facing ``name`` argument is an enum of the set's qualified
    ``<bundle>/<skill>`` names, so the model cannot ask for a skill that is not
    bound; with no skills, or more than ``MAX_ENUM_NAMES``, it is a plain
    string. An error reaches the model as a tool message with status
    ``error`` and a text it can act on; nothing is raised past the tool.
    A response is at most ``MAX_OUTPUT_CHARS`` long: a longer file is cut,
    and a list of names is cut at ``MAX_LIST_CHARS``. Non-text files are
    refused. Reading a ``SKILL.md`` records the SDK's ``Skill load`` span.

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
            raise ToolException(
                f"Unknown skill '{name}'. Available: {_name_list(names) or 'none'}"
            )
        bundle, skill = hit
        path = path.replace("\\", "/").removeprefix("./")
        try:
            text = skill_set.read(name, path)
        except KeyError as e:
            raise ToolException(
                f"Skill '{name}' has no file '{path}'. Files: {_name_list(skill.files)}"
            ) from e
        except RuntimeError as e:
            raise ToolException(f"Skill '{name}' is not available any more: {e}") from e
        except OSError as e:
            # strerror only: the exception's own text carries the local file path.
            raise ToolException(
                f"Skill '{name}' could not be read: {e.strerror or type(e).__name__}"
            ) from e
        if "\x00" in text[:_BINARY_SNIFF_CHARS]:
            raise ToolException(
                f"Skill '{name}' file '{path}' is not a text file and cannot be shown."
            )
        return _render(bundle, skill, path, text)

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
        handle_tool_error=True,
    )
