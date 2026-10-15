"""Refactoring prompts for Python code improvement."""

from .extract_method import extract_method
from .simplify_conditional import simplify_conditional
from .remove_duplication import remove_duplication
from .improve_naming import improve_naming

__all__ = [
    "extract_method",
    "simplify_conditional",
    "remove_duplication",
    "improve_naming",
]
