"""Exceptions for the basic agent loop."""

from typing import Any

from uipath.runtime.errors import UiPathRuntimeError


class AgentNodeRoutingException(Exception):
    pass


class AgentTerminationException(UiPathRuntimeError):
    pass


class AgentToolTerminationRequest(BaseException):
    """Tool requested agent termination (signal, not error).

    Inherits from BaseException (like SystemExit, StopIteration) to avoid
    accidental catching by `except Exception` blocks. This follows Python's
    EAFP idiom for flow control signals.

    Each loop implementation catches this and handles termination its own way:
    - LangGraph loop: UiPathToolNode catches -> Command(goto=TERMINATE)
    - Coded agents: catch -> terminate gracefully
    - Deterministic loops: catch -> return final result
    """

    def __init__(
        self,
        source: str,
        title: str,
        detail: str,
        output: Any = None,
    ):
        self.source = source
        self.title = title
        self.detail = detail
        self.output = output
        super().__init__(title)
