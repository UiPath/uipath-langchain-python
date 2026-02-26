"""Verify BTS components are properly importable and have expected interfaces."""

from uipath_agents._bts.bts_callback import BtsCallback
from uipath_agents._bts.bts_runtime import BtsRuntime
from uipath_agents._bts.bts_storage import SqliteBtsStateStorage


def test_bts_callback_is_async_callback_handler() -> None:
    from langchain_core.callbacks import AsyncCallbackHandler

    assert issubclass(BtsCallback, AsyncCallbackHandler)


def test_bts_runtime_has_delegate_property() -> None:
    assert hasattr(BtsRuntime, "delegate")


def test_bts_storage_namespace() -> None:
    assert hasattr(SqliteBtsStateStorage, "NAMESPACE")
    assert SqliteBtsStateStorage.NAMESPACE == "bts_state"
