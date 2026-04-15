"""Base class for built-in UiPath guardrail middlewares."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform import UiPath
from uipath.platform.guardrails import BuiltInValidatorGuardrail
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

from uipath_langchain.agent.exceptions import AgentRuntimeError

from ..models import GuardrailAction
from ._utils import convert_block_exception, extract_text_from_messages

logger = logging.getLogger(__name__)


class BuiltInGuardrailMiddlewareMixin:
    """Mixin providing shared evaluation logic for built-in guardrail middlewares.

    Subclasses must set:
        _guardrail (BuiltInValidatorGuardrail): The guardrail configuration.
        _name (str): The guardrail name used in log messages.
        action (GuardrailAction): The action to take on violation.
    """

    _guardrail: BuiltInValidatorGuardrail
    _name: str
    action: GuardrailAction
    _uipath: UiPath | None = None

    def _get_uipath(self) -> UiPath:
        """Get or create UiPath instance."""
        if self._uipath is None:
            self._uipath = UiPath()
        return self._uipath

    def _evaluate_guardrail(
        self, input_data: str | dict[str, Any]
    ) -> GuardrailValidationResult:
        """Evaluate the guardrail against input data via the UiPath API."""
        uipath = self._get_uipath()
        return uipath.guardrails.evaluate_guardrail(input_data, self._guardrail)

    def _handle_validation_result(
        self, result: GuardrailValidationResult, input_data: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Delegate to the action when a violation is detected."""
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            return self.action.handle_validation_result(result, input_data, self._name)
        return None

    def _check_messages(self, messages: list[BaseMessage]) -> None:
        """Evaluate guardrail against message text; apply action on violation."""
        if not messages:
            return

        text = extract_text_from_messages(messages)
        if not text:
            return

        try:
            result = self._evaluate_guardrail(text)
            modified_text = self._handle_validation_result(result, text)
            if (
                modified_text is not None
                and isinstance(modified_text, str)
                and modified_text != text
            ):
                for msg in messages:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        if isinstance(msg.content, str) and text in msg.content:
                            msg.content = msg.content.replace(text, modified_text, 1)
                            break
        except GuardrailBlockException as exc:
            raise convert_block_exception(exc) from exc
        except AgentRuntimeError:
            raise
        except Exception as e:
            logger.error(
                f"Error evaluating guardrail '{self._name}': {e}", exc_info=True
            )
