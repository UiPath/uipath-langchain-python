"""The ``load_skill`` tool: a coded agent's on-demand access to a bound skill's instructions."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model
from uipath.platform.skills import Skill, SkillBundle, SkillSet

_SKILL_MD = "SKILL.md"


def _render(bundle: SkillBundle, skill: Skill, path: str, text: str) -> str:
    """The tool output for one file: a header the model can cite, then the text."""
    qualified = f"{bundle.ref.name}/{skill.name}"
    if path == _SKILL_MD:
        others = [f for f in skill.files if f != _SKILL_MD]
        footer = f"\n\nSupporting files: {', '.join(others)}" if others else ""
        return f"# {qualified} (bundle {bundle.ref.name} v{bundle.version})\n\n{text}{footer}"
    return f"# {qualified} — {path}\n\n{text}"


def create_load_skill_tool(
    skill_set: SkillSet, *, name: str = "load_skill"
) -> StructuredTool:
    """A tool that returns a skill's ``SKILL.md`` (or one of its files) by qualified name.

    The ``name`` argument is an enum of the set's qualified ``<bundle>/<skill>``
    names, so the model cannot ask for a skill that is not bound; with no skills
    it is a plain string. Errors are returned as text so the model can recover;
    nothing is raised past the tool node. Reading a ``SKILL.md`` records the
    SDK's ``Skill load`` span.

    Args:
        skill_set: The skills bound to the agent, materialized to disk.
        name: The tool name the model sees; also its ``display_name``.

    Returns:
        A ``StructuredTool`` with sync and async implementations and
        ``metadata={"tool_type": "internal", "display_name": name}``.
    """
    names = skill_set.names()
    name_type: Any = Literal[tuple(names)] if names else str
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
        try:
            text = skill_set.read(name, path)
        except KeyError:
            return (
                f"Skill '{name}' has no file '{path}'. Files: {', '.join(skill.files)}"
            )
        except RuntimeError as e:
            return f"Skill '{name}' is not available any more: {e}"
        return _render(bundle, skill, path, text)

    async def aload(name: str, path: str = _SKILL_MD) -> str:
        return load(name, path)

    return StructuredTool.from_function(
        func=load,
        coroutine=aload,
        name=name,
        args_schema=args,
        description=(
            "Load the full instructions of one of the skills listed under "
            "'## Skills' before following it."
        ),
        metadata={"tool_type": "internal", "display_name": name},
    )
