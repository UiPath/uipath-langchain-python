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
        f"1. When a skill matches the task, call `{tool_name}` with its name and "
        "read the result before acting.\n"
        "2. To read a supporting file the skill mentions, call "
        f"`{tool_name}` with the same name and its `path`."
    )
