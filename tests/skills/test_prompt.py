"""Tests for the skills prompt section."""

import re

from uipath.platform.skills import SkillSet

from uipath_langchain.skills import (
    UIPATH_SKILLS_PROMPT,
    autonomous_skills_directive,
    skills_prompt_section,
)


def test_prompt_section_wraps_the_sdk_section(skill_set: SkillSet) -> None:
    text = skills_prompt_section(skill_set, tool_name="load_skill")
    assert text.startswith(skill_set.prompt_section() + "\n\n**How to use a skill**")
    assert (
        "1. When a skill matches the task, call `load_skill` with its name and "
        "read the result before acting." in text
    )
    assert (
        "2. To read a supporting file the skill mentions, call `load_skill` with "
        "the same name and its `path`." in text
    )
    assert "finance/invoice-triage" in text


def test_prompt_section_does_not_repeat_the_sdk_rules(skill_set: SkillSet) -> None:
    text = skills_prompt_section(skill_set, tool_name="load_skill")
    assert text.count("cannot be executed") == 1
    rules = text.split("**How to use a skill**", 1)[1]
    numbered = [line for line in rules.splitlines() if re.match(r"\d+\. ", line)]
    assert [line[:3] for line in numbered] == ["1. ", "2. "]


def test_prompt_section_uses_the_tool_name(skill_set: SkillSet) -> None:
    text = skills_prompt_section(skill_set, tool_name="read_skill")
    assert "call `read_skill` with its name" in text
    assert "call `read_skill` with the same name" in text
    assert "`load_skill`" not in text


def test_prompt_section_defaults_to_load_skill(skill_set: SkillSet) -> None:
    assert "call `load_skill` with its name" in skills_prompt_section(skill_set)


def test_prompt_section_empty_for_empty_set(empty_skill_set: SkillSet) -> None:
    assert skills_prompt_section(empty_skill_set, tool_name="load_skill") == ""


def test_template_has_the_three_deepagents_slots() -> None:
    for slot in ("{skills_locations}", "{skills_load_warnings}", "{skills_list}"):
        assert slot in UIPATH_SKILLS_PROMPT


def test_template_formats_without_other_placeholders() -> None:
    rendered = UIPATH_SKILLS_PROMPT.format(
        skills_locations="L", skills_load_warnings="", skills_list="S"
    )
    assert "{" not in rendered and "}" not in rendered


def test_template_says_scripts_are_reference_and_never_mentions_deepagents() -> None:
    assert (
        "Scripts inside a skill are reference material; do not run them unless "
        "your tools explicitly support it and the task requires it."
        in " ".join(UIPATH_SKILLS_PROMPT.split())
    )
    assert "cannot be executed" not in UIPATH_SKILLS_PROMPT
    assert (
        "Deepagents" not in UIPATH_SKILLS_PROMPT
        and "deepagents" not in UIPATH_SKILLS_PROMPT
    )
    assert "read_file" in UIPATH_SKILLS_PROMPT and "limit=1000" in UIPATH_SKILLS_PROMPT
    assert "`<bundle>/<skill>`" in UIPATH_SKILLS_PROMPT


def test_template_tells_how_to_resolve_paths_and_hand_skills_to_subagents() -> None:
    assert "relative to its folder" in UIPATH_SKILLS_PROMPT
    assert "/skills/<bundle>/<skill>/references/x.md" in UIPATH_SKILLS_PROMPT
    assert "Subagents you delegate to do not see this list" in UIPATH_SKILLS_PROMPT
    assert "pick by description" in UIPATH_SKILLS_PROMPT
    assert "continue with the tools you have" in UIPATH_SKILLS_PROMPT


def test_autonomous_directive_mentions_approval_gates() -> None:
    directive = autonomous_skills_directive()
    assert "approval" in directive
    assert "delete, disable, overwrite, or stop" in directive
    assert "start," not in directive
