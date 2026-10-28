"""Skill bundles for coded LangGraph agents.

``UiPathSkills`` declares an agent's bound bundles, resolves them once per
process and hands out the ``load_skill`` tool and the prompt section, or the
middleware for an advanced (deepagents) agent. ``create_load_skill_tool`` and
``skills_prompt_section`` are the building blocks for an already-resolved
``uipath.platform.skills.SkillSet``; ``create_skills_middleware``,
``SkillLoadTraceMiddleware`` and ``UIPATH_SKILLS_PROMPT`` are the
advanced-agent ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .load_skill_tool import create_load_skill_tool
from .prompt import (
    UIPATH_SKILLS_PROMPT,
    autonomous_skills_directive,
    skills_prompt_section,
)
from .uipath_skills import UiPathSkills

if TYPE_CHECKING:
    from .middleware import (
        SkillLoadTraceMiddleware,
        UiPathSkillsMiddleware,
        create_skills_middleware,
        deepagents_sources,
    )

__all__ = [
    "UIPATH_SKILLS_PROMPT",
    "SkillLoadTraceMiddleware",
    "UiPathSkills",
    "UiPathSkillsMiddleware",
    "autonomous_skills_directive",
    "create_load_skill_tool",
    "create_skills_middleware",
    "deepagents_sources",
    "skills_prompt_section",
]

# The middleware pulls deepagents and langchain's agent middleware; load it
# only when asked for, so the load_skill path never imports deepagents.
_LAZY_MIDDLEWARE_NAMES = frozenset(
    {
        "SkillLoadTraceMiddleware",
        "UiPathSkillsMiddleware",
        "create_skills_middleware",
        "deepagents_sources",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_MIDDLEWARE_NAMES:
        from . import middleware

        value = getattr(middleware, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
