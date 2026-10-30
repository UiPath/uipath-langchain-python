"""Tests for the skills middleware wiring over a real workspace backend."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware, SkillsState
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from uipath.platform.skills import SkillSet

from uipath_langchain.skills import UIPATH_SKILLS_PROMPT, UiPathSkills
from uipath_langchain.skills.middleware import (
    SkillLoadTraceMiddleware,
    UiPathSkillsMiddleware,
    create_skills_middleware,
    deepagents_sources,
)


@pytest.fixture
def workspace(skill_set: SkillSet) -> Path:
    """The backend root: ``skill_set`` lives in its ``skills`` folder.

    finance (invoice-triage, vendor-lookup) and ops (vendor-lookup): a name collision.
    """
    assert skill_set.root is not None and skill_set.root.name == "skills"
    return skill_set.root.parent


def _backend(workspace: Path) -> FilesystemBackend:
    return FilesystemBackend(root_dir=workspace, virtual_mode=True)


def _middleware(workspace: Path, skill_set: SkillSet) -> UiPathSkillsMiddleware:
    return create_skills_middleware(_backend(workspace), skill_set)


def test_deepagents_sources_are_one_labelled_root_per_bundle(
    skill_set: SkillSet,
) -> None:
    assert deepagents_sources(skill_set) == [
        ("/skills/finance/", "finance v3"),
        ("/skills/ops/", "ops v1"),
    ]


def test_deepagents_sources_empty_for_empty_set(empty_skill_set: SkillSet) -> None:
    assert deepagents_sources(empty_skill_set) == []


def test_builds_a_skills_middleware_with_one_source_per_bundle(
    workspace: Path, skill_set: SkillSet
) -> None:
    middleware = _middleware(workspace, skill_set)
    assert isinstance(middleware, UiPathSkillsMiddleware) and isinstance(
        middleware, SkillsMiddleware
    )
    assert middleware.sources == ["/skills/finance/", "/skills/ops/"]
    assert middleware.source_labels == ["finance v3", "ops v1"]
    assert middleware.system_prompt_template is UIPATH_SKILLS_PROMPT


def test_same_skill_name_in_two_bundles_keeps_both_under_qualified_names(
    workspace: Path, skill_set: SkillSet
) -> None:
    middleware = _middleware(workspace, skill_set)
    update = middleware.before_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    assert update is not None
    by_name = {s["name"]: s for s in update["skills_metadata"]}
    assert sorted(by_name) == [
        "finance/invoice-triage",
        "finance/vendor-lookup",
        "ops/vendor-lookup",
    ]
    assert by_name["finance/vendor-lookup"]["path"] == (
        "/skills/finance/vendor-lookup/SKILL.md"
    )
    assert by_name["ops/vendor-lookup"]["path"] == "/skills/ops/vendor-lookup/SKILL.md"
    assert by_name["ops/vendor-lookup"]["description"].startswith("(ops copy)")
    assert "skills_load_errors" not in update


async def test_async_listing_matches_sync(workspace: Path, skill_set: SkillSet) -> None:
    middleware = _middleware(workspace, skill_set)
    sync_update = middleware.before_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    async_update = await middleware.abefore_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    assert sync_update == async_update


def test_second_turn_does_not_relist(workspace: Path, skill_set: SkillSet) -> None:
    middleware = _middleware(workspace, skill_set)
    state: SkillsState = {"messages": [], "skills_metadata": []}
    assert middleware.before_agent(state, None, {}) is None  # type: ignore[arg-type]


async def test_async_second_turn_does_not_relist(
    workspace: Path, skill_set: SkillSet
) -> None:
    middleware = _middleware(workspace, skill_set)
    state: SkillsState = {"messages": [], "skills_metadata": []}
    assert await middleware.abefore_agent(state, None, {}) is None  # type: ignore[arg-type]


def test_missing_bundle_folder_is_reported_and_other_bundles_still_listed(
    workspace: Path, skill_set: SkillSet
) -> None:
    shutil.rmtree(workspace / "skills" / "finance")
    middleware = _middleware(workspace, skill_set)
    update = middleware.before_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    assert update is not None
    assert [s["name"] for s in update["skills_metadata"]] == ["ops/vendor-lookup"]
    errors = update.get("skills_load_errors", [])
    assert len(errors) == 1 and "/skills/finance/" in errors[0]


async def test_async_missing_bundle_folder_is_reported(
    workspace: Path, skill_set: SkillSet
) -> None:
    shutil.rmtree(workspace / "skills" / "finance")
    middleware = _middleware(workspace, skill_set)
    sync_update = middleware.before_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    async_update = await middleware.abefore_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    assert async_update == sync_update


def test_rendered_section_lists_qualified_names_and_paths(
    workspace: Path, skill_set: SkillSet
) -> None:
    middleware = _middleware(workspace, skill_set)
    update = middleware.before_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    assert update is not None
    text = UIPATH_SKILLS_PROMPT.format(
        skills_locations=middleware._format_skills_locations(),
        skills_load_warnings="",
        skills_list=middleware._format_skills_list(update["skills_metadata"]),
    )
    assert "## Skills" in text
    assert "**finance v3 Skills**: `/skills/finance/`" in text
    assert "- **finance/vendor-lookup**:" in text and "- **ops/vendor-lookup**:" in text
    assert "/skills/ops/vendor-lookup/SKILL.md" in text
    assert "cannot be executed in this environment" in " ".join(text.split())


def test_modify_request_appends_the_section_to_the_system_message(
    workspace: Path, skill_set: SkillSet
) -> None:
    middleware = _middleware(workspace, skill_set)
    update = middleware.before_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    assert update is not None
    state: SkillsState = {
        "messages": [],
        "skills_metadata": update["skills_metadata"],
    }
    request = ModelRequest(
        model=FakeListChatModel(responses=["ok"]),
        messages=[HumanMessage(content="hi")],
        system_message=SystemMessage(content="You are an agent."),
        state=state,
    )
    modified = middleware.modify_request(request)
    assert modified.system_message is not None
    text = modified.system_message.text
    assert text.startswith("You are an agent.")
    assert "## Skills" in text
    assert "- **ops/vendor-lookup**:" in text
    assert "/skills/finance/invoice-triage/SKILL.md" in text
    assert "**finance v3 Skills**: `/skills/finance/`" in text
    assert "**ops v1 Skills**: `/skills/ops/`" in text
    assert "higher priority" not in text


def _loaded(skill_set: SkillSet) -> UiPathSkills:
    skills = UiPathSkills([b.ref for b in skill_set.bundles])
    skills._skill_set = skill_set
    return skills


def test_advanced_middleware_requires_a_loaded_set(workspace: Path) -> None:
    skills = UiPathSkills([])
    with pytest.raises(RuntimeError, match="skills.load\\(destination="):
        skills.advanced_middleware(_backend(workspace))


def test_advanced_middleware_is_empty_without_skills(
    workspace: Path, empty_skill_set: SkillSet
) -> None:
    assert _loaded(empty_skill_set).advanced_middleware(_backend(workspace)) == []


def test_advanced_middleware_lists_and_traces_after_load(
    workspace: Path, skill_set: SkillSet
) -> None:
    middleware = _loaded(skill_set).advanced_middleware(_backend(workspace))
    assert [type(m) for m in middleware] == [
        UiPathSkillsMiddleware,
        SkillLoadTraceMiddleware,
    ]
    listing = middleware[0]
    assert isinstance(listing, UiPathSkillsMiddleware)
    update = listing.before_agent({"messages": []}, None, {})  # type: ignore[arg-type]
    assert update is not None
    assert len(update["skills_metadata"]) == 3


def test_advanced_middleware_warns_when_the_root_is_not_a_skills_folder(
    tmp_path: Path, skill_set: SkillSet, caplog: pytest.LogCaptureFixture
) -> None:
    elsewhere = skill_set.model_copy(update={"root": tmp_path / "bundles"})
    with caplog.at_level("WARNING", logger="uipath_langchain.skills.uipath_skills"):
        middleware = _loaded(elsewhere).advanced_middleware(_backend(tmp_path))
    assert len(middleware) == 2
    assert any("destination=<workspace>/skills" in r.message for r in caplog.records)


def test_advanced_middleware_rejects_a_cleaned_up_set(
    workspace: Path, skill_set: SkillSet
) -> None:
    gone = skill_set.model_copy(update={"root": None})
    with pytest.raises(RuntimeError, match="no folder"):
        _loaded(gone).advanced_middleware(_backend(workspace))


def test_importing_the_package_does_not_import_deepagents() -> None:
    code = (
        "import sys, uipath_langchain.skills as s\n"
        "assert 'deepagents' not in sys.modules, 'deepagents imported'\n"
        "assert 'uipath_langchain.skills.middleware' not in sys.modules\n"
        "assert not hasattr(s, 'create_skills_middleware')\n"
        "import uipath_langchain.skills.middleware\n"
        "assert 'deepagents' in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
