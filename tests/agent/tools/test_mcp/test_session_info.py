"""Tests for SessionInfo and SessionInfoFactory classes."""

from unittest.mock import MagicMock

import pytest

from uipath_langchain.agent.tools.mcp import SessionInfo, SessionInfoFactory


class TestSessionInfo:
    """Tests for base SessionInfo class."""

    @pytest.mark.asyncio
    async def test_default_session_id_is_none(self) -> None:
        info = SessionInfo()
        assert await info.get_session_id() is None

    @pytest.mark.asyncio
    async def test_init_with_session_id(self) -> None:
        info = SessionInfo(session_id="abc-123")
        assert await info.get_session_id() == "abc-123"

    @pytest.mark.asyncio
    async def test_set_session_id(self) -> None:
        info = SessionInfo()
        await info.set_session_id("new-id")
        assert await info.get_session_id() == "new-id"

    @pytest.mark.asyncio
    async def test_set_session_id_overwrites(self) -> None:
        info = SessionInfo(session_id="old")
        await info.set_session_id("new")
        assert await info.get_session_id() == "new"

    @pytest.mark.asyncio
    async def test_set_session_id_none_clears(self) -> None:
        info = SessionInfo(session_id="existing")
        await info.set_session_id(None)
        assert await info.get_session_id() is None

    @pytest.mark.asyncio
    async def test_session_id_attribute_matches(self) -> None:
        info = SessionInfo()
        await info.set_session_id("test")
        assert info.session_id == "test"


class TestSessionInfoFactory:
    """Tests for default SessionInfoFactory."""

    def test_creates_session_info(self) -> None:
        factory = SessionInfoFactory()
        server = MagicMock()
        server.slug = "my-server"
        server.folder_key = "folder-123"

        result = factory.create_session(server)
        assert isinstance(result, SessionInfo)

    @pytest.mark.asyncio
    async def test_created_session_has_no_id(self) -> None:
        factory = SessionInfoFactory()
        server = MagicMock()
        server.slug = "my-server"
        server.folder_key = "folder-123"

        result = factory.create_session(server)
        assert await result.get_session_id() is None

    def test_subclass_can_override(self) -> None:
        """Verify factory is designed for subclassing."""
        from uipath.platform.orchestrator.mcp import McpServer

        class CustomInfo(SessionInfo):
            pass

        class CustomFactory(SessionInfoFactory):
            def create_session(self, mcp_server: McpServer) -> CustomInfo:
                return CustomInfo(session_id="custom")

        factory = CustomFactory()
        server = MagicMock()
        server.slug = "s"
        server.folder_key = "f"

        result = factory.create_session(server)
        assert isinstance(result, CustomInfo)
        assert result.session_id == "custom"
