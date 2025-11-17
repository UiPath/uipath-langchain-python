"""Custom exceptions for the lowcode agent graph builder."""

from __future__ import annotations

from typing import Sequence

from pydantic_core import ErrorDetails


class ConfigurationError(Exception):
    """Raised when agent configuration is invalid."""

    pass


class InputValidationError(Exception):
    """Raised when input arguments don't match schema."""

    def __init__(
        self,
        message: str,
        validation_errors: Sequence[ErrorDetails] | None = None,
    ):
        self.validation_errors = validation_errors
        super().__init__(message)
