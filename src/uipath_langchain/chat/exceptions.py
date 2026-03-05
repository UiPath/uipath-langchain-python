from enum import Enum

from uipath.runtime.errors import UiPathBaseRuntimeError, UiPathErrorCategory


class ChatModelErrorCode(str, Enum):
    UNSUCCESSFUL_STOP_REASON = "UNSUCCESSFUL_STOP_REASON"


class ChatModelError(UiPathBaseRuntimeError):
    """Exception for chat model errors independent of the agent loop."""

    def __init__(
        self,
        code: ChatModelErrorCode,
        title: str,
        detail: str,
        category: UiPathErrorCategory = UiPathErrorCategory.UNKNOWN,
        status: int | None = None,
    ):
        super().__init__(
            code.value, title, detail, category, status, prefix="CHAT_MODEL"
        )
