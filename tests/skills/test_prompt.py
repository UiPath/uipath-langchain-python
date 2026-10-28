"""Tests for the skills prompt section."""

import re

from uipath.platform.skills import SkillSet

from uipath_langchain.skills import skills_prompt_section


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
