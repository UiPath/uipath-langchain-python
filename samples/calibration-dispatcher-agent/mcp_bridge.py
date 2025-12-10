# mcp_bridge.py
"""
MCP Bridge for UiPath RPA Workflow Integration

This module provides a safe async-to-sync bridge for calling MCP tools from
LangChain/LangGraph agents. It handles:
- MCP client session management with auto-reconnect
- Async/sync compatibility for LangGraph nodes
- Fallback to classic UiPath process invocation
- Tool discovery and invocation with proper error handling

For configuration, see config.py
"""
from __future__ import annotations

import json
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# Environment variables
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Import centralized configuration
import config

# UiPath SDK
from uipath.platform import UiPath

# MCP client + LangChain
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

# ---------- Logging ----------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT
)
log = logging.getLogger("mcp-bridge")

# ---------- Config ----------
USE_MCP = config.USE_MCP
MCP_SERVER_URL = config.MCP_SERVER_URL
FOLDER_PATH = config.UIPATH_FOLDER_PATH

# InArgument names from config
ARG_EMAIL = config.MCP_ARG_EMAIL
ARG_SLACK = config.MCP_ARG_SLACK
ARG_ENTITY = config.MCP_ARG_ENTITY


# ---------- UiPath SDK client ----------
_uipath_client: Optional[UiPath] = None

def get_uipath_client() -> UiPath:
    global _uipath_client
    if _uipath_client is None:
        _uipath_client = UiPath()
    return _uipath_client

def get_access_token_from_sdk() -> Optional[str]:
    try:
        client = get_uipath_client()
        api_client = getattr(client, "api_client", None)
        headers = getattr(api_client, "default_headers", {}) if api_client else {}
        auth = headers.get("Authorization", "")
        if isinstance(auth, str) and auth.startswith("Bearer "):
            token = auth.replace("Bearer ", "", 1)
            if token:
                return token
    except Exception as e:
        log.debug("Could not read token from SDK: %s", e)
    return None

def get_access_token() -> Optional[str]:
    return os.getenv("UIPATH_ACCESS_TOKEN") or get_access_token_from_sdk()


_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_thread: Optional[Thread] = None

def _ensure_bg_loop():
    global _bg_loop, _bg_thread
    if _bg_loop and _bg_loop.is_running():
        return
    _bg_loop = asyncio.new_event_loop()
    def _runner():
        asyncio.set_event_loop(_bg_loop)  # Required for anyio compatibility
        _bg_loop.run_forever()
    _bg_thread = Thread(target=_runner, name="mcp-bg-loop", daemon=True)
    _bg_thread.start()

def _run_coro_sync(coro):
    _ensure_bg_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, _bg_loop)  # type: ignore[arg-type]
    return fut.result()

# ---------- MCP session ----------
_http_cm = None  # async context manager transportu MCP
_session_read = None
_session_write = None
_transport = None
_session: Optional[ClientSession] = None
_tools_cache: List = []
_session_lock = asyncio.Lock()

async def _open_session() -> ClientSession:
    """Open new session mcp"""
    global _http_cm, _session_read, _session_write, _transport, _session
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if not MCP_SERVER_URL:
        raise RuntimeError("MCP_SERVER_URL is not set. Provide it from Orchestrator (MCP Servers).")

    
    _http_cm = streamablehttp_client(url=MCP_SERVER_URL, headers=headers, timeout=90)

    _session_read, _session_write, _transport = await _http_cm.__aenter__()
    try:
        sid = _transport.get_session_id() if hasattr(_transport, "get_session_id") else "unknown"
        log.info("Received session ID: %s", sid)
    except Exception:
        log.info("Received session ID: unknown")

    _session = ClientSession(_session_read, _session_write)
    await _session.__aenter__()
    await _session.initialize()
    log.info("Negotiated protocol version: %s", getattr(_session, "protocol_version", "unknown"))
    return _session

async def _ensure_session() -> ClientSession:
    global _session
    if _session is not None:
        return _session
    async with _session_lock:
        if _session is None:
            _session = await _open_session()
    return _session

async def _close_session():
    """Close session MCP"""
    global _http_cm, _session_read, _session_write, _transport, _session
    try:
        if _session is not None:
            try:
                await _session.__aexit__(None, None, None)
            finally:
                _session = None
        if _http_cm is not None:
            try:
                await _http_cm.__aexit__(None, None, None)
            finally:
                _http_cm = None
    finally:
        _session_read = None
        _session_write = None
        _transport = None

#  clean shutdown at exit
import atexit
def _shutdown_sync():
    try:
        if _bg_loop and _bg_loop.is_running():
            # close mcp session
            asyncio.run_coroutine_threadsafe(_close_session(), _bg_loop).result(timeout=3)
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)
            if _bg_thread:
                _bg_thread.join(timeout=2)
    except Exception:
        pass
atexit.register(_shutdown_sync)

async def get_mcp_tools(refresh: bool = False):
    """Download tools MCP"""
    global _tools_cache
    if refresh:
        _tools_cache = []
    if _tools_cache:
        return _tools_cache
    async with _session_lock:
        if _tools_cache:
            return _tools_cache
        session = await _ensure_session()
        _tools_cache = await load_mcp_tools(session)
        names = [t.name for t in _tools_cache]
        log.info("MCP tools discovered: %s", names)
        return _tools_cache

def _normalize(s: str) -> str:
    return (s or "").lower().replace(" ", "_")

async def _find_tool(name: str):
    tools = await get_mcp_tools()
    target = _normalize(name)
    for t in tools:
        if _normalize(t.name) == target:
            return t
    for t in tools:
        if target in _normalize(t.name):
            return t
    return None

async def _invoke_tool_once(tool_name: str, payload: Any, arg_name: str) -> Any:
    """
    Single try to invoke MCP tool
    """
    await _ensure_session()
    tool = await _find_tool(tool_name)
    if not tool:
        await get_mcp_tools(refresh=True)
        tool = await _find_tool(tool_name)
        if not tool:
            raise RuntimeError(
                f"MCP tool '{tool_name}' not found. "
                f"Exposed tools: {[t.name for t in await get_mcp_tools()]}"
            )

    payload_str = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    args = {arg_name: payload_str}

    log.info("Calling MCP tool '%s' with arg_name='%s', payload_len=%d chars",
             tool.name, arg_name, len(payload_str))
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Payload preview: %s", payload_str[:500])

    return await tool.ainvoke(args)

async def _invoke_tool(tool_name: str, payload: Any, arg_name: str) -> Any:
    """
    Invoke tool with auto-reconnect and 1 retry if stream is closed
    or "cancel scope/asyncgen" error occurs.
    """
    from anyio import ClosedResourceError
    try:
        return await _invoke_tool_once(tool_name, payload, arg_name)
    except (ClosedResourceError, RuntimeError, GeneratorExit) as e:
        msg = str(e)
        if isinstance(e, ClosedResourceError) or "cancel scope" in msg or "asynchronous generator" in msg:
            log.warning("MCP stream/context issue. Reconnecting... (%s)", msg or type(e).__name__)
            await _close_session()
            await _open_session()
            await get_mcp_tools(refresh=True)
            return await _invoke_tool_once(tool_name, payload, arg_name)
        raise

# ======================================================================
#                           PUBLIC API
# ======================================================================

def send_calibration_notifications_mcp(route_data: Dict[str, Any]) -> bool:
    async def _run():
        return await _invoke_tool("send_Calibration_Notifications", route_data, ARG_EMAIL)
    try:
        res = _run_coro_sync(_run())
        log.info("MCP email workflow completed: %s", str(res)[:200])
        return True
    except Exception as e:
        log.error("MCP email workflow failed: %s", e, exc_info=True)
        return False

def send_slack_notification_mcp(slack_payload: Dict[str, Any]) -> bool:
    async def _run():
        return await _invoke_tool("send_Slack_Notification", slack_payload, ARG_SLACK)
    try:
        res = _run_coro_sync(_run())
        log.info("MCP Slack workflow completed: %s", str(res)[:200])
        return True
    except Exception as e:
        log.error("MCP Slack workflow failed: %s", e, exc_info=True)
        return False

def add_service_order_mcp(record: Dict[str, Any]) -> bool:
    async def _run():
        return await _invoke_tool("addServiceOrder", record, ARG_ENTITY)
    try:
        res = _run_coro_sync(_run())
        log.info("MCP AddServiceOrder completed: %s", str(res)[:200])
        return True
    except Exception as e:
        log.error("MCP AddServiceOrder failed: %s", e, exc_info=True)
        return False

# ---------- Classic fallback (e-mail) ----------
def send_calibration_notifications_classic(route_data: Dict[str, Any]) -> bool:
    try:
        client = get_uipath_client()
        payload = json.dumps(route_data, ensure_ascii=False)
        res = client.processes.invoke(
            name="Send_Calibration_Notifications",
            folder_path=FOLDER_PATH or None,
            input_arguments={ARG_EMAIL: payload},
        )
        job_id = getattr(res, "id", None) or str(res)
        log.info("Classic invoke OK. Job: %s", job_id)
        return True
    except Exception as e:
        log.error("Classic invoke failed: %s", e, exc_info=True)
        return False

# ---------- Facade used by main ----------
def send_calibration_notifications(route_data: Dict[str, Any]) -> bool:
    return send_calibration_notifications_mcp(route_data) if USE_MCP else send_calibration_notifications_classic(route_data)

def send_slack_notification(payload: Dict[str, Any]) -> bool:
    return send_slack_notification_mcp(payload) if USE_MCP else False

def add_service_order(record: Dict[str, Any]) -> bool:
    return add_service_order_mcp(record) if USE_MCP else False
