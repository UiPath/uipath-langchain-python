"""Tests for ``UiPathSkills``."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath.platform import UiPath
from uipath.platform.skills import SkillBundleRef, SkillSet

from uipath_langchain.skills import UiPathSkills, create_load_skill_tool

REFS = [SkillBundleRef(name="finance"), SkillBundleRef(name="ops")]


def _sdk(result: SkillSet | None = None) -> MagicMock:
    sdk = MagicMock(spec=UiPath)
    sdk.skills.load_all_async = AsyncMock(return_value=result)
    sdk.skills.load_all = MagicMock(return_value=result)
    return sdk


def _schema(tool: StructuredTool) -> dict[str, Any]:
    args_schema = tool.args_schema
    assert isinstance(args_schema, type) and issubclass(args_schema, BaseModel)
    return args_schema.model_json_schema()


async def test_load_calls_the_sdk_once_and_caches(skill_set: SkillSet) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    assert skills.skill_set is None
    assert await skills.load() is skill_set
    assert await skills.load() is skill_set
    assert skills.skill_set is skill_set
    sdk.skills.load_all_async.assert_awaited_once_with(REFS, destination=None)


async def test_load_force_reloads(skill_set: SkillSet) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    await skills.load()
    await skills.load(force=True)
    assert sdk.skills.load_all_async.await_count == 2


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


async def test_prompt_section_is_empty_before_load(skill_set: SkillSet) -> None:
    skills = UiPathSkills(REFS, uipath_sdk=_sdk(skill_set), tool_name="read_skill")
    assert skills.prompt_section() == ""
    await skills.load()
    section = skills.prompt_section()
    assert section.startswith("## Skills")
    assert "`read_skill`" in section


async def test_tool_after_load_is_built_once(skill_set: SkillSet) -> None:
    skills = UiPathSkills(REFS, uipath_sdk=_sdk(skill_set))
    await skills.load()
    tool = skills.load_skill_tool()
    assert tool is not None
    assert skills.load_skill_tool() is tool
    assert _schema(tool)["properties"]["name"]["enum"] == skill_set.names()
    out = await tool.ainvoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup (bundle ops v1)")


async def test_lazy_tool_loads_on_first_async_call(skill_set: SkillSet) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    tool = skills.load_skill_tool()
    assert tool is not None
    assert isinstance(tool.name, str) and tool.name == "load_skill"
    assert "enum" not in _schema(tool)["properties"]["name"]
    sdk.skills.load_all_async.assert_not_awaited()
    out = await tool.ainvoke({"name": "ops/vendor-lookup"})
    expected = await create_load_skill_tool(skill_set).ainvoke(
        {"name": "ops/vendor-lookup"}
    )
    assert out == expected
    await tool.ainvoke({"name": "finance/invoice-triage"})
    sdk.skills.load_all_async.assert_awaited_once()
    assert skills.skill_set is skill_set


def test_lazy_tool_sync_call_without_a_running_loop(skill_set: SkillSet) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    tool = skills.load_skill_tool()
    assert tool is not None
    out = tool.invoke(
        {"name": "finance/invoice-triage", "path": "reference/vendors.md"}
    )
    assert out.startswith("# finance/invoice-triage — reference/vendors.md")
    sdk.skills.load_all_async.assert_awaited_once()
    sdk.skills.load_all.assert_not_called()


async def test_lazy_tool_sync_call_inside_a_running_loop_uses_load_all(
    skill_set: SkillSet,
) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    tool = skills.load_skill_tool()
    assert tool is not None
    out = tool.invoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup (bundle ops v1)")
    sdk.skills.load_all.assert_called_once()
    sdk.skills.load_all_async.assert_not_awaited()


def test_lazy_tool_has_the_real_tools_metadata_and_description(
    skill_set: SkillSet,
) -> None:
    skills = UiPathSkills(REFS, uipath_sdk=_sdk(skill_set), tool_name="read_skill")
    lazy = skills.load_skill_tool()
    real = create_load_skill_tool(skill_set, tool_name="read_skill")
    assert lazy is not None
    assert lazy.name == real.name == "read_skill"
    assert lazy.metadata == real.metadata
    assert lazy.description == real.description


async def test_no_tool_after_load_when_the_set_has_no_skills(
    empty_skill_set: SkillSet,
) -> None:
    skills = UiPathSkills(REFS, uipath_sdk=_sdk(empty_skill_set))
    lazy = skills.load_skill_tool()
    assert lazy is not None
    assert await lazy.ainvoke({"name": "x"}) == "Unknown skill 'x'. Available: none"
    assert skills.load_skill_tool() is None
    assert skills.prompt_section() == ""


async def test_load_never_raises_and_caches_the_failure(skill_set: SkillSet) -> None:
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=RuntimeError("boom"))
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    result = await skills.load()
    assert result.bundles == []
    assert [f.ref for f in result.failures] == REFS
    assert {f.reason for f in result.failures} == {"source_unavailable"}
    assert result.failures[0].detail == repr(RuntimeError("boom"))
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
    assert skills.load_sync() is result
    factory.assert_not_called()


async def test_empty_bundles_with_an_sdk_do_not_call_it() -> None:
    sdk = _sdk()
    skills = UiPathSkills([], uipath_sdk=sdk)
    await skills.load()
    sdk.skills.load_all_async.assert_not_awaited()


@pytest.mark.parametrize("bad", ["", "load skill", "a/b", "x" * 65])
def test_invalid_tool_name_is_rejected_at_construction(bad: str) -> None:
    with pytest.raises(ValueError, match="tool_name"):
        UiPathSkills(REFS, tool_name=bad)


async def test_cleanup_cleans_the_set_and_resets_the_cache(
    skill_set: SkillSet,
) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    await skills.load()
    with patch.object(SkillSet, "cleanup", autospec=True) as cleanup:
        skills.cleanup()
    cleanup.assert_called_once_with(skill_set)
    assert skills.skill_set is None
    assert skills.prompt_section() == ""
    skills.cleanup()  # nothing loaded: no-op


def test_load_sync_uses_load_all_and_caches(skill_set: SkillSet) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    assert skills.load_sync() is skill_set
    assert skills.load_sync() is skill_set
    sdk.skills.load_all.assert_called_once_with(REFS, destination=None)
    sdk.skills.load_all_async.assert_not_awaited()


def test_load_sync_never_raises() -> None:
    sdk = _sdk()
    sdk.skills.load_all = MagicMock(side_effect=OSError("disk"))
    result = UiPathSkills(REFS, uipath_sdk=sdk).load_sync()
    assert {f.reason for f in result.failures} == {"source_unavailable"}
