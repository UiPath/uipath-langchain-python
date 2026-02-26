"""Instant-resume value types for internal tools."""

from typing import Any

from uipath.platform.context_grounding.context_grounding_index import (
    ContextGroundingIndex,
)

from uipath_langchain.agent.tools.durable_interrupt import ResumeValue


class ReadyEphemeralIndex(ResumeValue):
    """An ephemeral index that is already ready (no wait needed).

    Returned from @durable_interrupt when the index finished ingestion
    before the interrupt fires.  The wrapper injects the model_dump()
    into the scratchpad resume list so subsequent resume cycles replay
    the same value without creating a trigger.
    """

    def __init__(self, index: ContextGroundingIndex):
        self.index = index

    @property
    def resume_value(self) -> Any:
        return self.index.model_dump()
