"""Prompt text for skills in a coded LangGraph agent.

``skills_prompt_section`` goes with the ``load_skill`` tool.
``UIPATH_SKILLS_PROMPT`` is the ``SkillsMiddleware`` template for an advanced
(deepagents) agent that reads skills with ``read_file``.
"""

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


UIPATH_SKILLS_PROMPT = """## Skills

Skills are packaged instructions bound to this agent, grouped in bundles. Each skill below is \
named `<bundle>/<skill>` and lists what it does and when to use it; its full instructions live \
in a `SKILL.md` file you read on demand. Skills with the same name in different bundles are \
different skills; pick by description.

{skills_locations}{skills_load_warnings}

**Available skills**

{skills_list}

**How to use a skill**

1. When the task matches a skill's description, read its instructions first: \
`read_file(file_path="<the SKILL.md path above>", limit=1000)`.
2. Follow them step by step. A skill may reference supporting files in its folder; read those \
with `read_file`. Paths inside a skill are relative to its folder: a reference to \
`references/x.md` in `/skills/<bundle>/<skill>/SKILL.md` means \
`/skills/<bundle>/<skill>/references/x.md`.
3. Scripts or executables inside a skill cannot be executed in this environment. Do not try to \
run them; use their content as reference only.
4. A skill may mention tools by name. If a named tool is missing, continue with the tools you \
have and note in your answer what was missing.
5. Skill folders are read-only. Write your own work elsewhere in the filesystem.
6. Subagents you delegate to do not see this list. When a delegated task is covered by a skill, \
put that skill's SKILL.md path in the task instructions so the subagent reads it first."""
"""``SkillsMiddleware`` system-prompt template; deepagents fills the three ``{...}`` slots."""

_AUTONOMOUS_SKILLS_DIRECTIVE = """\
**Skills without a human in the loop**
You run autonomously; no one can answer a question or approve a step. When a skill asks you to \
choose something (for example a framework, a folder, or a name), pick the default the skill \
implies and continue. When a skill requires the user's approval before a step, or a command \
would delete, disable, overwrite, or stop something that already exists in the tenant, do not \
run it: stop and report which step you skipped and why. Do not reach the same effect another \
way."""


def autonomous_skills_directive() -> str:
    """Skills guidance for an autonomous agent (a conversational agent can ask the user).

    Returns:
        The directive text to append to an autonomous agent's system prompt.
    """
    return _AUTONOMOUS_SKILLS_DIRECTIVE
