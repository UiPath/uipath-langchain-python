"""Tests for ``UiPathSkills``."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath.platform import UiPath
from uipath.platform.skills import SkillBundleRef, SkillSet

from uipath_langchain.skills import UiPathSkills

REFS = [SkillBundleRef(name="finance"), SkillBundleRef(name="ops")]


def _sdk(result: SkillSet | None = None) -> MagicMock:
    sdk = MagicMock(spec=UiPath)
    sdk.skills.load_all_async = AsyncMock(return_value=result)
    return sdk


def _schema(tool: StructuredTool) -> dict[str, Any]:
    args_schema = tool.args_schema
    assert isinstance(args_schema, type) and issubclass(args_schema, BaseModel)
    return args_schema.model_json_schema()


def _owned(skill_set: SkillSet, root: Path) -> SkillSet:
    """``skill_set`` pointing at ``root``, which it owns like a temp folder."""
    root.mkdir()
    return skill_set.model_copy(update={"root": root, "owns_root": True})


async def test_load_calls_the_sdk_once_and_caches(skill_set: SkillSet) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    assert skills.skill_set is None
    assert await skills.load() is skill_set
    assert await skills.load() is skill_set
    assert skills.skill_set is skill_set
    sdk.skills.load_all_async.assert_awaited_once_with(REFS, destination=None)


async def test_concurrent_first_loads_hit_the_sdk_once(skill_set: SkillSet) -> None:
    async def slow(*args: Any, **kwargs: Any) -> SkillSet:
        await asyncio.sleep(0.01)
        return skill_set

    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=slow)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    first, second = await asyncio.gather(skills.load(), skills.load())
    assert first is second is skill_set
    sdk.skills.load_all_async.assert_awaited_once()


async def test_destination_is_passed_through_and_overridable(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk, destination=tmp_path / "a")
    await skills.load()
    sdk.skills.load_all_async.assert_awaited_with(REFS, destination=tmp_path / "a")
    await skills.load(destination=tmp_path / "b", force=True)
    sdk.skills.load_all_async.assert_awaited_with(REFS, destination=tmp_path / "b")


async def test_force_reload_replaces_the_set_and_its_tool(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    old = _owned(skill_set, tmp_path / "old")
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=[old, skill_set])
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    await skills.load()
    old_tool = skills.load_skill_tool()
    assert await skills.load(force=True) is skill_set
    assert not (tmp_path / "old").exists() and old.root is None
    new_tool = skills.load_skill_tool()
    assert new_tool is not None and new_tool is not old_tool
    out = await new_tool.ainvoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup")


async def test_force_reload_keeps_a_root_the_old_set_does_not_own(
    skill_set: SkillSet,
) -> None:
    old = skill_set.model_copy()
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=[old, skill_set])
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    await skills.load()
    await skills.load(force=True)
    assert old.root is not None and old.root.exists()


async def test_prompt_section_is_empty_before_load(skill_set: SkillSet) -> None:
    skills = UiPathSkills(REFS, uipath_sdk=_sdk(skill_set), tool_name="read_skill")
    assert skills.prompt_section() == ""
    await skills.load()
    section = skills.prompt_section()
    assert section.startswith("## Skills")
    assert "`read_skill`" in section


def test_tool_before_load_raises() -> None:
    with pytest.raises(RuntimeError, match="skills.load\\(\\)"):
        UiPathSkills(REFS).load_skill_tool()


async def test_tool_after_load_is_built_once(skill_set: SkillSet) -> None:
    skills = UiPathSkills(REFS, uipath_sdk=_sdk(skill_set))
    await skills.load()
    tool = skills.load_skill_tool()
    assert tool is not None
    assert skills.load_skill_tool() is tool
    assert _schema(tool)["properties"]["name"]["enum"] == skill_set.names()
    out = await tool.ainvoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup (bundle ops v1)")


async def test_no_tool_when_the_set_has_no_skills(empty_skill_set: SkillSet) -> None:
    skills = UiPathSkills(REFS, uipath_sdk=_sdk(empty_skill_set))
    await skills.load()
    assert skills.load_skill_tool() is None
    assert skills.prompt_section() == ""


async def test_load_never_raises_and_caches_the_failure(
    skill_set: SkillSet, caplog: pytest.LogCaptureFixture
) -> None:
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=RuntimeError("boom"))
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    with caplog.at_level("WARNING", logger="uipath_langchain.skills.uipath_skills"):
        result = await skills.load()
    assert result.bundles == []
    assert [f.ref for f in result.failures] == REFS
    assert {f.reason for f in result.failures} == {"source_unavailable"}
    assert result.failures[0].detail == repr(RuntimeError("boom"))
    [record] = caplog.records
    assert record.exc_info is not None and isinstance(record.exc_info[1], RuntimeError)
    assert await skills.load() is result
    assert sdk.skills.load_all_async.await_count == 1
    sdk.skills.load_all_async = AsyncMock(return_value=skill_set)
    assert await skills.load(force=True) is skill_set


async def test_sdk_construction_error_becomes_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = MagicMock(side_effect=ValueError("no credentials"))
    monkeypatch.setattr("uipath_langchain.skills.uipath_skills.UiPath", factory)
    skills = UiPathSkills(REFS)
    factory.assert_not_called()
    result = await skills.load()
    assert {f.reason for f in result.failures} == {"source_unavailable"}
    assert "no credentials" in result.failures[0].detail
    factory.assert_called_once()


async def test_empty_bundles_never_touch_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = MagicMock()
    monkeypatch.setattr("uipath_langchain.skills.uipath_skills.UiPath", factory)
    skills = UiPathSkills([])
    result = await skills.load()
    assert result == SkillSet(source="none", bundles=[], failures=[])
    assert skills.load_skill_tool() is None
    factory.assert_not_called()


async def test_cleanup_removes_the_set_and_the_next_load_resolves_again(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    owned = _owned(skill_set, tmp_path / "owned")
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=[owned, skill_set])
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    await skills.load()
    skills.cleanup()
    assert not (tmp_path / "owned").exists() and owned.root is None
    assert skills.skill_set is None
    assert skills.prompt_section() == ""
    skills.cleanup()  # nothing loaded: no-op
    assert await skills.load() is skill_set
