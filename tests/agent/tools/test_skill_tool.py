"""Tests for skill_tool.py."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from uipath.agent.models.agent import (
    AgentSkillToolProperties,
    AgentSkillToolResourceConfig,
    AgentToolType,
)
from uipath.platform.skills import Skill, SkillVersion, SkillVersionStatus

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.agent.tools.skill_tool import create_skill_tool

SKILL_ID = "11111111-1111-1111-1111-111111111111"
VERSION_ID = "22222222-2222-2222-2222-222222222222"
PROMPT = "You are a skill that classifies tickets."


def _make_version(content: str = PROMPT) -> SkillVersion:
    return SkillVersion.model_validate(
        {
            "Id": VERSION_ID,
            "SkillId": SKILL_ID,
            "Version": "1.0.0",
            "Content": content,
            "Status": int(SkillVersionStatus.PUBLISHED),
            "PublishedAt": datetime.now(timezone.utc).isoformat(),
            "CreatedDate": datetime.now(timezone.utc).isoformat(),
        }
    )


def _make_skill(with_published: bool = True) -> Skill:
    payload: dict[str, Any] = {
        "Id": SKILL_ID,
        "Name": "classify-ticket",
        "Description": None,
        "GracePeriodDays": 30,
        "CreatedDate": datetime.now(timezone.utc).isoformat(),
        "LastUpdatedDate": datetime.now(timezone.utc).isoformat(),
        "FolderKey": "33333333-3333-3333-3333-333333333333",
        "PublishedVersion": None,
        "CurrentDraft": None,
        "Versions": [],
        "Tags": [],
    }
    if with_published:
        payload["PublishedVersion"] = {
            "Id": VERSION_ID,
            "Version": "1.0.0",
            "Status": int(SkillVersionStatus.PUBLISHED),
            "PublishedAt": datetime.now(timezone.utc).isoformat(),
            "CreatedDate": datetime.now(timezone.utc).isoformat(),
        }
    return Skill.model_validate(payload)


@pytest.fixture
def skill_resource() -> AgentSkillToolResourceConfig:
    return AgentSkillToolResourceConfig(
        type=AgentToolType.SKILL,
        name="classify-ticket",
        description="Classify a support ticket",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_schema={
            "type": "object",
            "properties": {"category": {"type": "string"}},
            "required": ["category"],
        },
        properties=AgentSkillToolProperties(skill_id=SKILL_ID),
    )


@pytest.fixture
def skill_resource_pinned() -> AgentSkillToolResourceConfig:
    return AgentSkillToolResourceConfig(
        type=AgentToolType.SKILL,
        name="classify-ticket",
        description="Classify a support ticket",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        output_schema={"type": "object", "properties": {}},
        properties=AgentSkillToolProperties(skill_id=SKILL_ID, version_id=VERSION_ID),
    )


def _llm_mock(structured_return: Any = None, text_return: str = "ok") -> MagicMock:
    """Build a BaseChatModel mock with model_copy + ainvoke + with_structured_output."""
    llm = MagicMock()
    llm_copy = MagicMock()
    llm.model_copy.return_value = llm_copy

    response_msg = MagicMock()
    response_msg.content_blocks = [{"type": "text", "text": text_return}]
    llm_copy.ainvoke = AsyncMock(return_value=response_msg)

    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=structured_return)
    llm_copy.with_structured_output.return_value = structured_llm
    return llm


class TestSkillToolStartup:
    def test_resolves_published_version_when_unpinned(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        sdk.skills.get_version.return_value = _make_version()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource, llm=_llm_mock())
        sdk.skills.retrieve.assert_called_once_with(
            key=SKILL_ID, folder_path=None, folder_key=None
        )
        sdk.skills.get_version.assert_called_once_with(
            key=SKILL_ID,
            version_id=VERSION_ID,
            folder_path=None,
            folder_key=None,
        )
        assert tool.metadata is not None
        assert tool.metadata["tool_type"] == "skill"
        assert tool.metadata["skill_id"] == SKILL_ID
        assert tool.metadata["skill_version_id"] is None

    def test_resolves_pinned_version_directly(
        self, skill_resource_pinned: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.get_version.return_value = _make_version()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource_pinned, llm=_llm_mock())
        sdk.skills.retrieve.assert_not_called()
        sdk.skills.get_version.assert_called_once_with(
            key=SKILL_ID,
            version_id=VERSION_ID,
            folder_path=None,
            folder_key=None,
        )
        assert tool.metadata is not None
        assert tool.metadata["skill_version_id"] == VERSION_ID

    def test_raises_when_skill_has_no_published_version(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill(with_published=False)
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            with pytest.raises(AgentRuntimeError) as exc_info:
                create_skill_tool(skill_resource, llm=_llm_mock())
        assert "no published version" in exc_info.value.error_info.title.lower()

    def test_raises_when_skill_not_found(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.side_effect = LookupError("Skill 'X' not found")
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            with pytest.raises(AgentRuntimeError) as exc_info:
                create_skill_tool(skill_resource, llm=_llm_mock())
        assert "not found" in exc_info.value.error_info.title.lower()


class TestSkillToolInvocation:
    @pytest.mark.asyncio
    async def test_calls_llm_with_system_prompt_and_args(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        sdk.skills.get_version.return_value = _make_version()

        structured_result = MagicMock()
        structured_result.model_dump.return_value = {"category": "billing"}
        llm = _llm_mock(structured_return=structured_result)

        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource, llm=llm)

        result = await tool.coroutine(text="My invoice is wrong")

        assert result == {"category": "billing"}
        structured_llm = llm.model_copy.return_value.with_structured_output.return_value
        messages = structured_llm.ainvoke.await_args.args[0]
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == PROMPT
        assert isinstance(messages[1], HumanMessage)
        assert "My invoice is wrong" in messages[1].content

    @pytest.mark.asyncio
    async def test_falls_back_to_text_when_no_output_schema(
        self, skill_resource_pinned: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.get_version.return_value = _make_version()
        llm = _llm_mock(text_return="42")

        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource_pinned, llm=llm)

        result = await tool.coroutine(text="anything")

        # Output schema has no properties, so fallback path runs and JSON-parses
        # the plain text response.
        assert result == 42
        # with_structured_output should NOT have been invoked
        llm.model_copy.return_value.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_fallback_returns_string_when_not_json(
        self, skill_resource_pinned: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.get_version.return_value = _make_version()
        llm = _llm_mock(text_return="hello world")

        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource_pinned, llm=llm)

        result = await tool.coroutine(text="anything")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_llm_failure_wraps_in_agent_runtime_error(
        self, skill_resource: AgentSkillToolResourceConfig
    ):
        sdk = MagicMock()
        sdk.skills.retrieve.return_value = _make_skill()
        sdk.skills.get_version.return_value = _make_version()
        llm = _llm_mock()
        llm.model_copy.return_value.with_structured_output.return_value.ainvoke = (
            AsyncMock(side_effect=RuntimeError("LLM down"))
        )

        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            tool = create_skill_tool(skill_resource, llm=llm)

        with pytest.raises(AgentRuntimeError):
            await tool.coroutine(text="x")


class TestSkillToolFolderScoping:
    def test_passes_folder_args_through(self):
        resource = AgentSkillToolResourceConfig(
            type=AgentToolType.SKILL,
            name="classify-ticket",
            description="Classify a support ticket",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            properties=AgentSkillToolProperties(
                skill_id=SKILL_ID,
                version_id=VERSION_ID,
                folder_path="/Shared/MyFolder",
                folder_key="folder-key-1",
            ),
        )
        sdk = MagicMock()
        sdk.skills.get_version.return_value = _make_version()
        with patch("uipath_langchain.agent.tools.skill_tool.UiPath", return_value=sdk):
            create_skill_tool(resource, llm=_llm_mock())
        sdk.skills.get_version.assert_called_once_with(
            key=SKILL_ID,
            version_id=VERSION_ID,
            folder_path="/Shared/MyFolder",
            folder_key="folder-key-1",
        )
