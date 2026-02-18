"""Value matchers and format validators for trace assertions."""

import re
from typing import Any


class FormatValidator:
    """Validates attribute values against predefined formats."""

    UUID_PATTERN = re.compile(
        r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        re.IGNORECASE,
    )
    ISO_DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    HEX_ID_PATTERN = re.compile(r"^[a-f0-9]+$", re.IGNORECASE)

    @classmethod
    def validate(cls, value: Any, format_type: str) -> bool:
        """Validate value against format type.

        Args:
            value: Value to validate.
            format_type: One of "uuid", "iso_datetime", "hex_id", "non_empty".

        Returns:
            True if valid, False otherwise.
        """
        if format_type == "uuid":
            return bool(cls.UUID_PATTERN.match(str(value)))
        elif format_type == "iso_datetime":
            return bool(cls.ISO_DATETIME_PATTERN.match(str(value)))
        elif format_type == "hex_id":
            return bool(cls.HEX_ID_PATTERN.match(str(value)))
        elif format_type == "non_empty":
            return bool(value) and str(value).strip() != ""
        else:
            raise ValueError(f"Unknown format type: {format_type}")


class ValueMatcher:
    """Matches attribute values against expected patterns."""

    @classmethod
    def matches(cls, actual: Any, expected: Any) -> bool:
        """Check if actual value matches expected pattern.

        Args:
            actual: Actual value from trace.
            expected: Expected value or pattern.
                - "*" matches any non-None value
                - list matches if actual is in list
                - otherwise exact match

        Returns:
            True if matches, False otherwise.
        """
        if expected == "*":
            return actual is not None
        elif isinstance(expected, list):
            return actual in expected
        else:
            return actual == expected
