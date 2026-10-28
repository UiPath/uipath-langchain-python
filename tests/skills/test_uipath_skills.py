"""Tests for ``UiPathSkills``."""

import asyncio
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


def _slow_sync(result: SkillSet) -> MagicMock:
    def slow(*args: Any, **kwargs: Any) -> SkillSet:
        time.sleep(0.05)
        return result

    return MagicMock(side_effect=slow)


def _slow_async(result: SkillSet) -> AsyncMock:
    async def slow(*args: Any, **kwargs: Any) -> SkillSet:
        await asyncio.sleep(0.05)
        return result

    return AsyncMock(side_effect=slow)


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
    sdk.skills.load_all.assert_called_once()
    sdk.skills.load_all_async.assert_not_awaited()


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


def test_bundles_are_the_declared_refs() -> None:
    skills = UiPathSkills(REFS)
    assert skills.bundles == tuple(REFS)


def test_load_sync_passes_destination_when_forced(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    sdk = _sdk(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk, destination=tmp_path / "a")
    skills.load_sync()
    sdk.skills.load_all.assert_called_with(REFS, destination=tmp_path / "a")
    skills.load_sync(destination=tmp_path / "b")
    assert sdk.skills.load_all.call_count == 1  # cached: destination unused
    skills.load_sync(destination=tmp_path / "b", force=True)
    sdk.skills.load_all.assert_called_with(REFS, destination=tmp_path / "b")


def test_two_threads_invoking_the_lazy_tool_resolve_once(skill_set: SkillSet) -> None:
    sdk = _sdk()
    sdk.skills.load_all = _slow_sync(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    tool = skills.load_skill_tool()
    assert tool is not None
    barrier = threading.Barrier(2)

    def call(_: int) -> str:
        barrier.wait()
        return tool.invoke({"name": "ops/vendor-lookup"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        outs = list(pool.map(call, range(2)))
    assert all(out.startswith("# ops/vendor-lookup") for out in outs)
    assert sdk.skills.load_all.call_count == 1


def test_load_and_load_sync_in_two_threads_resolve_once(skill_set: SkillSet) -> None:
    sdk = _sdk()
    sdk.skills.load_all = _slow_sync(skill_set)
    sdk.skills.load_all_async = _slow_async(skill_set)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    barrier = threading.Barrier(2)

    def run_async() -> SkillSet:
        barrier.wait()
        return asyncio.run(skills.load())

    def run_sync() -> SkillSet:
        barrier.wait()
        return skills.load_sync()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(run_async), pool.submit(run_sync)]
        assert all(r.result() is skill_set for r in results)
    assert sdk.skills.load_all.call_count + sdk.skills.load_all_async.await_count == 1


async def test_load_sync_on_the_loop_running_load_resolves_separately(
    skill_set: SkillSet,
) -> None:
    started = asyncio.Event()

    async def slow(*args: Any, **kwargs: Any) -> SkillSet:
        started.set()
        await asyncio.sleep(0.05)
        return skill_set

    sdk = _sdk(skill_set)
    sdk.skills.load_all_async = AsyncMock(side_effect=slow)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    task = asyncio.create_task(skills.load())
    await started.wait()
    # Blocking on the in-flight load() here would deadlock this loop.
    assert skills.load_sync() is skill_set
    sdk.skills.load_all.assert_called_once()
    assert await task is skill_set


async def test_cancelled_owner_lets_a_waiter_resolve(skill_set: SkillSet) -> None:
    started = asyncio.Event()
    calls: list[int] = []

    async def side(*args: Any, **kwargs: Any) -> SkillSet:
        calls.append(1)
        if len(calls) == 1:
            started.set()
            await asyncio.sleep(10)
        return skill_set

    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=side)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    owner = asyncio.create_task(skills.load())
    await started.wait()
    waiter = asyncio.create_task(skills.load())
    await asyncio.sleep(0)
    owner.cancel()
    assert await waiter is skill_set
    assert len(calls) == 2
    with pytest.raises(asyncio.CancelledError):
        await owner


async def test_force_reload_removes_the_old_owned_temp_root(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    old_root = tmp_path / "old"
    old_root.mkdir()
    (old_root / "marker").write_text("x")
    old = skill_set.model_copy(update={"root": old_root, "owns_root": True})
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=[old, skill_set])
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    await skills.load()
    old_tool = skills.load_skill_tool()
    assert old_tool is not None
    assert await skills.load(force=True) is skill_set
    assert not old_root.exists()
    assert old.root is None
    new_tool = skills.load_skill_tool()
    assert new_tool is not None and new_tool is not old_tool
    out = await new_tool.ainvoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup")
    stale = await old_tool.ainvoke({"name": "ops/vendor-lookup"})
    assert stale.startswith("Skill 'ops/vendor-lookup' is not available any more")


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


async def test_lazy_tool_follows_a_forced_reload(
    skill_set: SkillSet, empty_skill_set: SkillSet
) -> None:
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=[empty_skill_set, skill_set])
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    lazy = skills.load_skill_tool()
    assert lazy is not None
    before = await lazy.ainvoke({"name": "ops/vendor-lookup"})
    assert before == "Unknown skill 'ops/vendor-lookup'. Available: none"
    await skills.load(force=True)
    after = await lazy.ainvoke({"name": "ops/vendor-lookup"})
    assert after.startswith("# ops/vendor-lookup (bundle ops v1)")


async def test_lazy_tool_resolves_again_after_cleanup(skill_set: SkillSet) -> None:
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(
        side_effect=lambda *args, **kwargs: skill_set.model_copy()
    )
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    lazy = skills.load_skill_tool()
    assert lazy is not None
    await lazy.ainvoke({"name": "ops/vendor-lookup"})
    skills.cleanup()
    assert skills.skill_set is None
    out = await lazy.ainvoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup (bundle ops v1)")
    assert sdk.skills.load_all_async.await_count == 2


async def test_failure_is_logged_with_the_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=RuntimeError("boom"))
    with caplog.at_level("WARNING", logger="uipath_langchain.skills.uipath_skills"):
        await UiPathSkills(REFS, uipath_sdk=sdk).load()
    [record] = caplog.records
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], RuntimeError)


def _owned_copy(skill_set: SkillSet, root: Path) -> SkillSet:
    """``skill_set`` copied into ``root``, which the copy owns (as a temp folder)."""
    assert skill_set.root is not None
    shutil.copytree(skill_set.root, root)
    return skill_set.model_copy(update={"root": root, "owns_root": True})


async def test_same_loop_load_sync_resolves_into_a_fresh_temp_folder(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(*args: Any, **kwargs: Any) -> SkillSet:
        started.set()
        await release.wait()
        return skill_set

    sdk = _sdk(skill_set)
    sdk.skills.load_all_async = AsyncMock(side_effect=slow)
    shared = tmp_path / "shared"
    skills = UiPathSkills(REFS, uipath_sdk=sdk, destination=shared)
    task = asyncio.create_task(skills.load())
    await started.wait()
    skills.load_sync()
    # Never the folder the in-flight owner is writing.
    sdk.skills.load_all.assert_called_once_with(REFS, destination=None)
    release.set()
    assert await task is skill_set
    sdk.skills.load_all_async.assert_awaited_once_with(REFS, destination=shared)


async def test_lazy_tool_cleans_up_the_uncached_same_loop_set(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(*args: Any, **kwargs: Any) -> SkillSet:
        started.set()
        await release.wait()
        return skill_set

    separate = _owned_copy(skill_set, tmp_path / "separate")
    sdk = _sdk(separate)
    sdk.skills.load_all_async = AsyncMock(side_effect=slow)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    tool = skills.load_skill_tool()
    assert tool is not None
    task = asyncio.create_task(skills.load())
    await started.wait()
    out = tool.invoke({"name": "ops/vendor-lookup"})
    assert out.startswith("# ops/vendor-lookup")
    sdk.skills.load_all.assert_called_once()
    assert not (tmp_path / "separate").exists()
    assert separate.root is None
    assert skills._uncached == []
    release.set()
    assert await task is skill_set
    assert skills.skill_set is skill_set
    assert skill_set.root is not None and skill_set.root.exists()


async def test_cleanup_during_a_resolution_discards_its_result(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    result = _owned_copy(skill_set, tmp_path / "owned")

    async def slow(*args: Any, **kwargs: Any) -> SkillSet:
        started.set()
        await release.wait()
        return result

    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=slow)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    task = asyncio.create_task(skills.load())
    await started.wait()
    skills.cleanup()
    release.set()
    assert await task is result
    assert skills.skill_set is None
    assert not (tmp_path / "owned").exists()
    assert result.root is None
    # The next load resolves again and is cached as usual.
    sdk.skills.load_all_async = AsyncMock(return_value=skill_set)
    assert await skills.load() is skill_set
    assert skills.skill_set is skill_set


async def test_cleanup_removes_the_uncached_same_loop_sets(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(*args: Any, **kwargs: Any) -> SkillSet:
        started.set()
        await release.wait()
        return skill_set

    separate = _owned_copy(skill_set, tmp_path / "separate")
    sdk = _sdk(separate)
    sdk.skills.load_all_async = AsyncMock(side_effect=slow)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    task = asyncio.create_task(skills.load())
    await started.wait()
    assert skills.load_sync() is separate
    release.set()
    assert await task is skill_set
    assert (tmp_path / "separate").exists()
    skills.cleanup()
    assert not (tmp_path / "separate").exists()
    assert separate.root is None


class _HookLock:
    """A lock that runs ``hook`` once, right after the first release once armed."""

    def __init__(self, hook: Callable[[], None]) -> None:
        self._inner = threading.Lock()
        self._hook = hook
        self.armed = False

    def __enter__(self) -> "_HookLock":
        self._inner.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.release()
        if self.armed:
            self.armed = False
            self._hook()


async def test_cleanup_right_after_the_finish_decision_is_not_lost(
    skill_set: SkillSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _owned_copy(skill_set, tmp_path / "owned")
    skills = UiPathSkills(REFS, uipath_sdk=_sdk())
    lock = _HookLock(skills.cleanup)
    monkeypatch.setattr(skills, "_lock", lock)

    async def arm_then_return(*args: Any, **kwargs: Any) -> SkillSet:
        # The next lock release is the first one inside _finish.
        lock.armed = True
        return result

    assert skills._sdk is not None
    skills._sdk.skills.load_all_async = AsyncMock(side_effect=arm_then_return)
    assert await skills.load() is result
    assert skills.skill_set is None
    assert not (tmp_path / "owned").exists()


async def test_waiter_of_a_stale_resolution_claims_again(
    skill_set: SkillSet, tmp_path: Path
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    first = _owned_copy(skill_set, tmp_path / "first")
    calls: list[int] = []

    async def side(*args: Any, **kwargs: Any) -> SkillSet:
        calls.append(1)
        if len(calls) == 1:
            started.set()
            await release.wait()
            return first
        return skill_set

    sdk = _sdk()
    sdk.skills.load_all_async = AsyncMock(side_effect=side)
    skills = UiPathSkills(REFS, uipath_sdk=sdk)
    owner = asyncio.create_task(skills.load())
    await started.wait()
    waiter = asyncio.create_task(skills.load())
    await asyncio.sleep(0)
    assert skills._inflight is not None
    skills.cleanup()
    release.set()
    assert await owner is first
    assert first.root is None
    assert await waiter is skill_set
    assert len(calls) == 2
    assert skills.skill_set is skill_set
