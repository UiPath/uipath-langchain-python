"""Reporter runtime that defers startup errors to the execution path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator

from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathStreamOptions,
)
from uipath.runtime.events import UiPathRuntimeEvent
from uipath.runtime.result import UiPathRuntimeResult
from uipath.runtime.schema import UiPathRuntimeSchema

if TYPE_CHECKING:
    from uipath.agent.models.agent import AgentDefinition


class ReporterRuntime:
    """Runtime that re-raises a captured error during execution.

    Allows debug infrastructure to report startup errors
    by deferring them to the normal execution path.
    This allows SignalR events, telemetry and spans to be handled out of the box
    """

    def __init__(
        self,
        error: Exception,
        agent_definition: AgentDefinition | None = None,
    ) -> None:
        self._error = error
        self.agent_definition = agent_definition

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        raise self._error

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        raise self._error
        yield  # noqa: unreachable

    async def get_schema(self) -> UiPathRuntimeSchema:
        raise self._error

    async def dispose(self) -> None:
        pass
