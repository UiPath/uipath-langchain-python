"""Tests for the skills prompt section."""

from uipath.platform.skills import SkillSet

from uipath_langchain.skills import skills_prompt_section


def test_prompt_section_mentions_tool_and_rules(skill_set: SkillSet) -> None:
    text = skills_prompt_section(skill_set, tool_name="load_skill")
    assert text.startswith("## Skills")
    assert text.startswith(skill_set.prompt_section())
    assert "`load_skill`" in text
    assert "cannot be executed" in text
    assert "**How to use a skill**" in text
    assert "finance/invoice-triage" in text
    assert "ops/vendor-lookup" in text


def test_prompt_section_uses_the_tool_name(skill_set: SkillSet) -> None:
    text = skills_prompt_section(skill_set, tool_name="read_skill")
    assert "`read_skill`" in text
    assert "`load_skill`" not in text


def test_prompt_section_defaults_to_load_skill(skill_set: SkillSet) -> None:
    assert "`load_skill`" in skills_prompt_section(skill_set)


def test_prompt_section_empty_for_empty_set(empty_skill_set: SkillSet) -> None:
    assert skills_prompt_section(empty_skill_set, tool_name="load_skill") == ""
