"""Skill bundles for coded LangGraph agents.

``UiPathSkills`` declares an agent's bound bundles, resolves them once per
process and hands out the ``load_skill`` tool and the prompt section, or the
middleware for an advanced (deepagents) agent. ``create_load_skill_tool`` and
``skills_prompt_section`` are the building blocks for an already-resolved
``uipath.platform.skills.SkillSet``; ``UIPATH_SKILLS_PROMPT`` is the
advanced-agent one.

The advanced-agent middleware (``create_skills_middleware``,
``SkillLoadTraceMiddleware``, ``UiPathSkillsMiddleware``,
``deepagents_sources``) lives in ``uipath_langchain.skills.middleware``. It
imports deepagents, so it is not imported here and the ``load_skill`` path
never loads deepagents.
"""

from .load_skill_tool import create_load_skill_tool
from .prompt import (
    UIPATH_SKILLS_PROMPT,
    autonomous_skills_directive,
    skills_prompt_section,
)
from .uipath_skills import UiPathSkills

__all__ = [
    "UIPATH_SKILLS_PROMPT",
    "UiPathSkills",
    "autonomous_skills_directive",
    "create_load_skill_tool",
    "skills_prompt_section",
]
