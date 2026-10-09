"""The provider-specific half of the classifier tool.

A classifier tool answers typed questions about an input with probabilities. The
tool configuration and argument handling are shared (see ``tool.py``); a provider
supplies what differs between classification APIs: the name of the input
argument, the question models, how a request is built and sent, and how the
answers are checked.
"""

import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Mapping

from pydantic import TypeAdapter
from uipath.agent.models.agent import ClassifierProvider, ClassifierQuestion
from uipath.llm_client import UiPathAPIError
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

# A provider rejects an input it cannot take (invalid, or over its context length).
INPUT_ERROR_STATUSES = frozenset({400, 413, 422})

CONFIDENCE_SCHEMA: dict[str, Any] = {
    "type": "number",
    "description": "Confidence in the answer, from 0 to 1.",
}


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class ClassifierProviderAdapter(ABC):
    """What a classification API needs from the classifier tool.

    Answers are returned keyed by question name, with the fields the provider
    returns for their type.
    """

    provider: ClassVar[ClassifierProvider]
    # The name used in error messages, e.g. "Jev".
    display_name: ClassVar[str]
    # The tool argument holding what is classified.
    input_argument: ClassVar[str]
    # An optional tool argument holding images (job attachments), when supported.
    images_argument: ClassVar[str | None] = None
    question_types: ClassVar[type[Enum]]
    question_adapter: ClassVar[TypeAdapter[Any]]
    # The fields a builder question of each type has, besides its name and type.
    builder_fields: ClassVar[Mapping[Any, tuple[str, ...]]]
    # Stand-ins for the fields of a built question that are not static, so a
    # static field can be checked with the question models before any call.
    placeholder_fields: ClassVar[Mapping[str, Any]]
    # One answer of the output of questions only known at run time.
    generic_answer_schema: ClassVar[dict[str, Any]]

    @property
    def invalid_response_title(self) -> str:
        return f"Invalid {self.display_name} response"

    def invalid_response_error(self, detail: str) -> AgentRuntimeError:
        return AgentRuntimeError(
            code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
            title=self.invalid_response_title,
            detail=detail,
            category=UiPathErrorCategory.SYSTEM,
        )

    def missing_answer_error(
        self, question: ClassifierQuestion, detail: str
    ) -> AgentRuntimeError:
        return self.invalid_response_error(f"Question '{question.name}': {detail}")

    @abstractmethod
    def create_client(self, model: str) -> Any:
        """Create the client for ``model`` (an AgentRuntimeError when it fails)."""

    @abstractmethod
    async def ask(
        self,
        client: Any,
        input_value: Any,
        questions: list[ClassifierQuestion],
        images: list[str],
    ) -> Any:
        """Send the questions and return the decoded response.

        ``images`` are data URLs, empty unless ``images_argument`` is set.
        """

    @abstractmethod
    def convert_answers(
        self, questions: list[ClassifierQuestion], response: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Return the answer to each question keyed by question name, once checked."""

    @abstractmethod
    def provider_message(self, error: UiPathAPIError) -> str:
        """The provider's own message in an error response."""

    @abstractmethod
    def is_unknown_model_error(self, error: UiPathAPIError) -> bool:
        """Whether the provider rejected the call because the model does not exist."""
