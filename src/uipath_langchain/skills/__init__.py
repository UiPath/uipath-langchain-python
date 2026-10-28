"""Skill bundles for coded LangGraph agents.

``create_load_skill_tool`` gives the model on-demand access to a bound skill's
instructions; ``skills_prompt_section`` is the matching system-prompt text.
Both work on the SDK's ``uipath.platform.skills.SkillSet``.
"""

from .load_skill_tool import create_load_skill_tool
from .prompt import skills_prompt_section

__all__ = ["create_load_skill_tool", "skills_prompt_section"]
