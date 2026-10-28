"""Prompt text for skills in a coded LangGraph agent."""

from __future__ import annotations

from uipath.platform.skills import SkillSet


def skills_prompt_section(skill_set: SkillSet, *, tool_name: str = "load_skill") -> str:
    """The SDK's ``## Skills`` section plus how to use the ``tool_name`` tool.

    Args:
        skill_set: The skills bound to the agent.
        tool_name: The name the model knows the load-skill tool by; see
            ``create_load_skill_tool``.

    Returns:
        A markdown section to append to the system prompt, or ``""`` when no
        bundle has a skill.
    """
    base = skill_set.prompt_section()
    if not base:
        return ""
    return (
        f"{base}\n\n"
        "**How to use a skill**\n\n"
        f"1. When the task matches a skill's description, call `{tool_name}` with "
        "its name and read the result before acting.\n"
        f"2. A skill may reference supporting files; call `{tool_name}` with the "
        "same name and the file `path` to read one.\n"
        "3. Scripts inside a skill cannot be executed here; use their content as "
        "reference only.\n"
        "4. A skill may name tools you do not have; continue with the tools you "
        "have and note what was missing."
    )
