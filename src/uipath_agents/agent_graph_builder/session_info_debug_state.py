"""SessionInfo subclass that persists session IDs to AgentHub debug state."""

import logging
import os
from enum import Enum, auto
from urllib.parse import quote

import httpx
from uipath._utils._ssl_context import get_httpx_client_kwargs
from uipath.platform.orchestrator.mcp import McpServer
from uipath_langchain.agent.tools.mcp import SessionInfo, SessionInfoFactory

logger = logging.getLogger(__name__)


class _SessionState(Enum):
    """State machine for session ID lifecycle.

    NotLoaded → Loaded: on first get_session_id (loads from debug state)
    Loaded → Cleared: on set_session_id(None)
    Cleared → Loaded: on set_session_id(value)
    """

    NOT_LOADED = auto()
    LOADED = auto()
    CLEARED = auto()


class SessionInfoDebugState(SessionInfo):
    """SessionInfo that persists to AgentHub debug state.

    Uses a three-state machine to manage the session ID lifecycle:

    * **NotLoaded** — initial state; ``get_session_id`` loads from the
      debug-state endpoint and transitions to Loaded.
    * **Loaded** — session ID is known (may be None if debug state had
      nothing); ``get_session_id`` returns the cached value.
    * **Cleared** — session was explicitly cleared via
      ``set_session_id(None)``; ``get_session_id`` returns None without
      attempting to reload from debug state.

    All required context is read from environment variables:

    * ``UIPATH_URL`` – base URL (``https://cloud.uipath.com/org/tenant``)
    * ``UIPATH_ACCESS_TOKEN`` – bearer token
    * ``UIPATH_PROJECT_ID`` – agent ID (GUID)
    """

    def __init__(self, slug: str, folder_key: str) -> None:
        super().__init__()
        self._slug = slug
        self._folder_key = folder_key
        self._state = _SessionState.NOT_LOADED

    @property
    def key(self) -> str:
        """Debug-state key for this MCP resource."""
        return f"mcpsession:{self._folder_key}:{self._slug}"

    # -- async interface -----------------------------------------------------

    async def get_session_id(self) -> str | None:
        """Return session ID, lazily loading from debug state on first call.

        In Cleared state, returns None without attempting to reload.
        """
        if self._state == _SessionState.NOT_LOADED:
            self._state = _SessionState.LOADED
            stored = await self._load_from_debug_state()
            if stored is not None:
                self.session_id = stored
                logger.info(f"Loaded session ID from debug state: {stored}")
            else:
                logger.info("No session ID found in debug state")
        elif self._state == _SessionState.CLEARED:
            return None
        return self.session_id

    async def set_session_id(self, session_id: str | None) -> None:
        """Store session ID locally and persist to debug state.

        Setting to None transitions to Cleared state.
        Setting to a value transitions to Loaded state.
        """
        if session_id is None:
            self.session_id = None
            self._state = _SessionState.CLEARED
            logger.info("Session ID cleared")
        else:
            self.session_id = session_id
            self._state = _SessionState.LOADED
            await self._save_to_debug_state(session_id)
            logger.info(f"Session ID set and persisted: {session_id}")

    # -- debug-state HTTP helpers --------------------------------------------

    def _debug_state_url(self) -> str | None:
        base_url = os.getenv("UIPATH_URL")
        agent_id = os.getenv("UIPATH_PROJECT_ID")
        if not base_url or not agent_id:
            return None
        encoded_key = quote(self.key, safe="")
        return f"{base_url}/agenthub_/design/debugstate/{agent_id}/{encoded_key}"

    def _auth_headers(self) -> dict[str, str]:
        token = os.getenv("UIPATH_ACCESS_TOKEN", "")
        return {"Authorization": f"Bearer {token}"}

    async def _load_from_debug_state(self) -> str | None:
        url = self._debug_state_url()
        if url is None:
            return None
        try:
            async with httpx.AsyncClient(
                headers=self._auth_headers(), **get_httpx_client_kwargs()
            ) as client:
                response = await client.get(url)
                if response.status_code == 200:
                    return response.text
        except Exception:
            logger.debug("Failed to load session ID from debug state", exc_info=True)
        return None

    async def _save_to_debug_state(self, session_id: str) -> None:
        url = self._debug_state_url()
        if url is None:
            return
        try:
            async with httpx.AsyncClient(
                headers=self._auth_headers(), **get_httpx_client_kwargs()
            ) as client:
                await client.put(
                    url,
                    content=session_id,
                    headers={"Content-Type": "text/plain"},
                )
        except Exception:
            logger.debug("Failed to save session ID to debug state", exc_info=True)


class SessionInfoDebugStateFactory(SessionInfoFactory):
    """Factory that creates ``SessionInfoDebugState`` instances."""

    def create_session(self, mcp_server: McpServer) -> SessionInfoDebugState:
        """Create a SessionInfoDebugState from an McpServer."""
        logger.info(
            f"Creating debug state session for server '{mcp_server.slug}' "
            f"in folder '{mcp_server.folder_key}'"
        )
        return SessionInfoDebugState(
            slug=mcp_server.slug or "",
            folder_key=mcp_server.folder_key or "",
        )
