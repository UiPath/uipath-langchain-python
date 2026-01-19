"""Exceptions for the agent graph."""

from uipath.runtime.errors import UiPathRuntimeError


class AgentNodeRoutingException(Exception):
    pass


class AgentTerminationException(UiPathRuntimeError):
    pass


class AgentStateException(Exception):
    pass
