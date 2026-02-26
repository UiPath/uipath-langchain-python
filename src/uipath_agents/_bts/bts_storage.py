"""SQLite-based BTS state storage for suspend/resume persistence.

Adapts SqliteResumableStorage for BTS state persistence,
using the key-value storage interface from uipath-langchain-python.
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from uipath_langchain.runtime.storage import SqliteResumableStorage


class SqliteBtsStateStorage:
    """Adapts SqliteResumableStorage for BTS state persistence.

    Uses the async key-value storage interface (set_value/get_value)
    to persist BTS state across process boundaries during suspend/resume.
    """

    NAMESPACE = "bts_state"
    KEY = "bts_context"

    def __init__(self, storage: "SqliteResumableStorage") -> None:
        self._storage = storage

    async def save_bts_state(self, runtime_id: str, state_dict: dict[str, Any]) -> None:
        """Save BTS state for later resume."""
        await self._storage.set_value(
            runtime_id=runtime_id,
            namespace=self.NAMESPACE,
            key=self.KEY,
            value=state_dict,
        )

    async def load_bts_state(self, runtime_id: str) -> Optional[dict[str, Any]]:
        """Load previously saved BTS state."""
        return await self._storage.get_value(
            runtime_id=runtime_id,
            namespace=self.NAMESPACE,
            key=self.KEY,
        )

    async def clear_bts_state(self, runtime_id: str) -> None:
        """Remove BTS state after successful completion."""
        await self._storage.set_value(
            runtime_id=runtime_id,
            namespace=self.NAMESPACE,
            key=self.KEY,
            value=None,
        )
