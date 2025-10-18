from typing import Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from uipath._cli._runtime._contracts import UiPathRuntimeContext


class LangGraphRuntimeContext(UiPathRuntimeContext):
    """Context information passed throughout the runtime execution."""

    memory: Optional[AsyncSqliteSaver] = None
