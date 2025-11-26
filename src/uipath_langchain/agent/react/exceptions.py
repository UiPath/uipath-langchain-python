"""Exceptions for the basic agent loop."""

from uipath.runtime.errors import UiPathRuntimeError


class AgentNodeRoutingException(Exception):
    pass


class AgentTerminationException(UiPathRuntimeError):
    pass
