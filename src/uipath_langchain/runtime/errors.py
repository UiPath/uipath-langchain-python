from enum import Enum
from typing import Union

from uipath.runtime.errors import (
    UiPathBaseRuntimeError,
    UiPathErrorCategory,
    UiPathErrorCode,
)


class LangGraphErrorCode(Enum):
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"

    GRAPH_NOT_FOUND = "GRAPH_NOT_FOUND"
    GRAPH_IMPORT_ERROR = "GRAPH_IMPORT_ERROR"
    GRAPH_TYPE_ERROR = "GRAPH_TYPE_ERROR"
    GRAPH_VALUE_ERROR = "GRAPH_VALUE_ERROR"
    GRAPH_LOAD_ERROR = "GRAPH_LOAD_ERROR"
    GRAPH_INVALID_UPDATE = "GRAPH_INVALID_UPDATE"
    GRAPH_EMPTY_INPUT = "GRAPH_EMPTY_INPUT"

    DB_QUERY_FAILED = "DB_QUERY_FAILED"
    DB_TABLE_CREATION_FAILED = "DB_TABLE_CREATION_FAILED"
    HITL_EVENT_CREATION_FAILED = "HITL_EVENT_CREATION_FAILED"
    DB_INSERT_FAILED = "DB_INSERT_FAILED"
    LICENSE_NOT_AVAILABLE = "LICENSE_NOT_AVAILABLE"


class LangGraphRuntimeError(UiPathBaseRuntimeError):
    """Custom exception for LangGraph runtime errors with structured error information."""

    def __init__(
        self,
        code: Union[LangGraphErrorCode, UiPathErrorCode],
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.UNKNOWN,
        status: int | None = None,
    ):
        super().__init__(
            code.value, title, detail, category, status, prefix="LANGGRAPH"
        )
