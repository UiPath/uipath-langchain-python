"""Skill bundles for coded LangGraph agents.

``UiPathSkills`` declares an agent's bound bundles, resolves them once per
process and hands out the ``load_skill`` tool and the prompt section.
``create_load_skill_tool`` and ``skills_prompt_section`` are the building
blocks for an already-resolved ``uipath.platform.skills.SkillSet``.
"""

from .load_skill_tool import create_load_skill_tool
from .prompt import skills_prompt_section
from .uipath_skills import UiPathSkills

__all__ = ["UiPathSkills", "create_load_skill_tool", "skills_prompt_section"]
