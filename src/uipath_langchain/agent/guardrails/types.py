from enum import Enum


class ExecutionStage(str, Enum):
    """Execution stage enumeration."""

    PRE_EXECUTION = "preExecution"
    POST_EXECUTION = "postExecution"
