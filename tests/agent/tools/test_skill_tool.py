"""Tests for skill_tool.py.

The skill tool is a thin "load this prompt" wrapper: it fetches the published
Content once at startup, takes no arguments, and returns the Content string.
The parent agent's LLM treats the returned prompt as additional context.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentSkillToolProperties,
    AgentSkillToolResourceConfig,
    AgentToolType,
)
from uipath.platform.skills import Skill, SkillVersionStatus

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.skill_tool import create_skill_tool

SKILL_ID = "11111111-1111-1111-1111-111111111111"
VERSION_ID = "22222222-2222-2222-2222-222222222222"
PROMPT = "You are a skill that classifies tickets."


def _make_skill(with_published: bool = True, content: str = PROMPT) -> Skill:
    payload: dict[str, Any] = {
        "id": SKILL_ID,
        "name": "classify-ticket",
        "description": None,
        "gracePeriodDays": 30,
        "createdDate": datetime.now(timezone.utc).isoformat(),
        "lastUpdatedDate": datetime.now(timezone.utc).isoformat(),
        "folderKey": "33333333-3333-3333-3333-333333333333",
        "publishedVersion": None,
        "currentDraft": None,
        "versions": [],
        "tags": [],
    }
    if with_published:
        payload["publishedVersion"] = {
            "id": VERSION_ID,
            "version": "1.0.0",
            "status": SkillVersionStatus.PUBLISHED.value,
            "publishedAt": datetime.now(timezone.utc).isoformat(),
            "createdDate": datetime.now(timezone.utc).isoformat(),
        }
        payload["versions"] = [
            {
                "id": VERSION_ID,
                "skillId": SKILL_ID,
                "version": "1.0.0",
                "content": content,
                "status": SkillVersionStatus.PUBLISHED.value,
                "publishedAt": datetime.now(timezone.utc).isoformat(),
                "createdDate": datetime.now(timezone.utc).isoformat(),
            }
        ]
    return Skill.model_validate(payload)


@pytest.fixture
def skill_resource() -> AgentSkillToolResourceConfig:
    return AgentSkillToolResourceConfig(
        type=AgentToolType.SKILL,
        name="classify-ticket",
        description="Classify a support ticket",
        properties=AgentSkillToolProperties(skill_id=SKILL_ID, version_id=VERSION_ID),
    )


@pytest.fixture
def skill_resource_pinned() -> AgentSkillToolResourceConfig:
    return AgentSkillToolResourceConfig(
        type=AgentToolType.SKILL,
        name="classify-ticket",
        description="Classify a support ticket",
        properties=AgentSkillToolProperties(skill_id=SKILL_ID, version_id=VERSION_ID),
    )


class TestSkillToolStartup:
    def test_resolves_published_version_when_unpinned(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource)
        sdk.skills.retrieve.assert_called_once_with(
            key=SKILL_ID,
            include_content=True,
            folder_path=None,
            folder_key=None,
        )
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "skill"
        assert tool.metadata["skill_id"] == SKILL_ID
        assert tool.metadata["skill_version_id"] == VERSION_ID

    def test_pinned_version_resolution(
        self, skill_resource_pinned: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource_pinned)
        sdk.skills.retrieve.assert_called_once_with(
            key=SKILL_ID,
            include_content=True,
            folder_path=None,
            folder_key=None,
        )
        assert tool.metadata is not None
        assert tool.metadata["skill_version_id"] == VERSION_ID

    def test_raises_when_skill_not_found(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.side_effect = LookupError("Skill 'X' not found")
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            with pytest.raises(AgentRuntimeError) as exc_info:
                create_skill_tool(skill_resource)
        assert "not found" in exc_info.value.error_info.title.lower()

    def test_raises_when_pinned_version_id_not_in_skill_versions(
        self, skill_resource_pinned: AgentSkillToolResourceConfig
    ):
        skill = _make_skill()
        skill.versions = []
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = skill
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            with pytest.raises(AgentRuntimeError) as exc_info:
                create_skill_tool(skill_resource_pinned)
        assert "no version" in exc_info.value.error_info.title.lower()


class TestSkillToolInvocation:
    @pytest.mark.asyncio
    async def test_returns_prompt_verbatim(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource)
        result = await tool.coroutine()
        assert result == PROMPT

    def test_tool_advertises_empty_args_schema(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource)
        schema = tool.args_schema.model_json_schema()
        assert schema.get("properties", {}) == {}


class TestSkillToolFolderScoping:
    def test_passes_folder_args_through(self):
        resource = AgentSkillToolResourceConfig(
            type=AgentToolType.SKILL,
            name="classify-ticket",
            description="Classify a support ticket",
            properties=AgentSkillToolProperties(
                skill_id=SKILL_ID,
                version_id=VERSION_ID,
                folder_path="/Shared/MyFolder",
                folder_key="folder-key-1",
            ),
        )
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            create_skill_tool(resource)
        sdk.skills.retrieve.assert_called_once_with(
            key=SKILL_ID,
            include_content=True,
            folder_path="/Shared/MyFolder",
            folder_key="folder-key-1",
        )
