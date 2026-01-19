"""Models for UiPath guardrails configuration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from uipath.core.guardrails import GuardrailValidationResult


@dataclass
class PIIDetectionEntity:
    """PII entity configuration with threshold.

    Args:
        name: The entity type (e.g., PIIDetectionEntity.EMAIL)
        threshold: Confidence threshold (0.0 to 1.0) for detection
    """

    name: str
    threshold: float = 0.5

    def __post_init__(self):
        """Validate threshold range."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0.0 and 1.0, got {self.threshold}"
            )


class GuardrailAction(ABC):
    """Interface for defining custom actions when guardrails are triggered.

    Extend this interface to implement custom behavior when guardrail validation fails.
    Common use cases include:
    - Logging violations (see actions.LogAction)
    - Blocking execution (see actions.BlockAction)
    - Escalating to monitoring systems
    - Sending alerts or notifications
    - Collecting metrics or analytics

    Example:
        ```python
        from uipath_langchain.guardrails import GuardrailAction
        from uipath.core.guardrails import GuardrailValidationResult, GuardrailValidationResultType

        class CustomAction(GuardrailAction):
            def handle_validation_result(
                self,
                result: GuardrailValidationResult,
                input_data: str | dict[str, Any],
                guardrail_name: str,
            ) -> str | dict[str, Any] | None:
                if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
                    # Your custom logic here
                    # Return modified data or None
                    return None
        ```
    """

    @abstractmethod
    def handle_validation_result(
        self,
        result: GuardrailValidationResult,
        data: str | dict[str, Any],
        guardrail_name: str,
    ) -> str | dict[str, Any] | None:
        """Handle a guardrail validation result.

        This method is called when a guardrail validation fails.
        Actions can optionally return modified data to sanitize/filter
        the validated data before execution continues.

        Args:
            result: The validation result from the guardrails service
            data: The data that was validated (string or dictionary).
                This can be tool input (arguments), tool output (result),
                or message content depending on the guardrail scope.
            guardrail_name: The name of the guardrail that triggered

        Returns:
            Modified data if the action wants to sanitize/filter the validated data,
            or None if no modification is needed. If None is returned, original
            data is used. If a value is returned, it replaces the original data.

            Note: The returned data type should match the data type:
            - For tool input: return dict[str, Any] (tool arguments)
            - For tool output: return dict[str, Any] (tool result)
            - For messages: return str (message content)
        """
        pass
