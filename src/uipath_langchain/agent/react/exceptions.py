"""Exceptions for the basic agent loop."""

from uipath._cli._runtime._contracts import UiPathRuntimeError


class AgentNodeRoutingException(Exception):
    pass


class AgentTerminationException(UiPathRuntimeError):
    pass
