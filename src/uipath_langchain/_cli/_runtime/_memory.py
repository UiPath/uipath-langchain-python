import os
from contextlib import asynccontextmanager

from uipath._cli._runtime._contracts import UiPathRuntimeContext

from ..._checkpointers.blob_sqlite_saver import AsyncBlobSqliteSaver


def get_connection_string(context: UiPathRuntimeContext) -> str:
    if context.runtime_dir and context.state_file:
        path = os.path.join(context.runtime_dir, context.state_file)
        if not context.resume and context.job_id is None:
            # If not resuming and no job id, delete the previous state file
            if os.path.exists(path):
                os.remove(path)
        os.makedirs(context.runtime_dir, exist_ok=True)
        return path
    return os.path.join("__uipath", "state.db")


@asynccontextmanager
async def get_memory(context: UiPathRuntimeContext):
    """Create and manage the AsyncSqliteSaver instance."""
    async with AsyncBlobSqliteSaver.from_filesystem(
        sqlite_path=get_connection_string(context),
        storage_path=".uipath/checkpointers/",
        job_guid=context.job_id,
    ) as memory:
        yield memory
