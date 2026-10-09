"""OpenAI's Decisions API as a classifier provider.

The Decisions API answers typed questions (``choice``, ``score``, ``predicate``)
about an ``input`` (text, optionally with images) with probabilities. Request and
response follow ``POST /v1/decisions``:
https://developers.openai.com/api/docs/guides/decisions
"""

import json
from typing import Any, ClassVar, Mapping, Sequence

from langchain_core.tools import ToolException
from pydantic import TypeAdapter
from uipath.agent.models.agent import (
    ClassifierProvider,
    ClassifierQuestion,
    DecisionsChoiceQuestion,
    DecisionsPredicateQuestion,
    DecisionsQuestion,
    DecisionsQuestionType,
    DecisionsScoreQuestion,
)
from uipath.llm_client import UiPathAPIError
from uipath.llm_client.clients.decisions import UiPathDecisionsClient
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

from .provider import (
    CONFIDENCE_SCHEMA,
    ClassifierProviderAdapter,
    is_finite_number,
)

# Decisions answers in well under a second; images make the request larger.
DECISIONS_TIMEOUT_SECONDS = 60.0

_QUESTION_TYPES = [question_type.value for question_type in DecisionsQuestionType]

_CHOICE_PROBABILITIES_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "Probability of each choice.",
    "items": {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "probability": {"type": "number"},
        },
    },
}

_SCORE_PROBABILITIES_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "Probability of each level, from lowest (0) to highest.",
    "items": {
        "type": "object",
        "properties": {
            "value": {"type": "integer", "description": "The level index."},
            "label": {"type": "string"},
            "probability": {"type": "number"},
        },
    },
}

# One answer of the generic output (Prompt, Argument): the fields the Decisions API
# returns for every type, without the name (the key of the answer).
_GENERIC_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": list(_QUESTION_TYPES),
            "description": "The question type.",
        },
        "choice": {"type": "string", "description": "The selected value (choice)."},
        "score": {
            "type": "number",
            "description": (
                "Probability-weighted level index, from 0 (lowest level) up; can "
                "land between levels (score)."
            ),
        },
        "confidence": CONFIDENCE_SCHEMA,
        "probabilities": {
            "type": "array",
            "description": (
                "Probability of each choice (choice) or of each level (score)."
            ),
            "items": {"type": "object"},
        },
        "probability": {
            "type": "number",
            "description": (
                "Probability that the condition is true, from 0 to 1 (predicate)."
            ),
        },
    },
    "required": ["type"],
    "additionalProperties": True,
}

# The fields every answer of a type has, each a number except ``choice``.
_ANSWER_FIELDS: dict[DecisionsQuestionType, tuple[str, ...]] = {
    DecisionsQuestionType.CHOICE: ("choice", "confidence"),
    DecisionsQuestionType.SCORE: ("score", "confidence"),
    DecisionsQuestionType.PREDICATE: ("probability",),
}


def _described(entry: dict[str, Any], description: str | None) -> dict[str, Any]:
    return {**entry, "description": description} if description else entry


def build_decisions_questions(
    questions: Sequence[ClassifierQuestion],
) -> list[dict[str, Any]]:
    """Translate validated Decisions questions into the ``questions`` payload."""
    payload: list[dict[str, Any]] = []
    for question in questions:
        entry: dict[str, Any] = {
            "type": question.type.value,
            "name": question.name,
            "instructions": question.instructions,
        }
        if isinstance(question, DecisionsChoiceQuestion):
            entry["choices"] = [
                _described({"value": choice.value}, choice.description)
                for choice in question.choices
            ]
        elif isinstance(question, DecisionsScoreQuestion):
            entry["levels"] = [
                _described({"label": level.label}, level.description)
                for level in question.levels
            ]
        payload.append(entry)
    return payload


def build_decisions_input(input_value: Any, images: list[str]) -> str | list[Any]:
    """The Decisions ``input``: the text, or one user message with text and images.

    An object or a list is sent as its JSON text.
    """
    text = (
        input_value
        if isinstance(input_value, str)
        else json.dumps(input_value, ensure_ascii=False)
    )
    if not images:
        return text
    content: list[dict[str, Any]] = (
        [{"type": "input_text", "text": text}] if text.strip() else []
    )
    content.extend({"type": "input_image", "image_url": image} for image in images)
    return [{"role": "user", "content": content}]


class DecisionsProvider(ClassifierProviderAdapter):
    """OpenAI's Decisions API, through the LLM Gateway's OpenAI passthrough."""

    provider: ClassVar[ClassifierProvider] = ClassifierProvider.OPENAI
    display_name: ClassVar[str] = "OpenAI Decisions"
    input_argument: ClassVar[str] = "input"
    images_argument: ClassVar[str | None] = "images"
    question_types: ClassVar[type[DecisionsQuestionType]] = DecisionsQuestionType
    question_adapter: ClassVar[TypeAdapter[Any]] = TypeAdapter(DecisionsQuestion)
    builder_fields: ClassVar[Mapping[Any, tuple[str, ...]]] = {
        DecisionsQuestionType.CHOICE: ("instructions", "choices"),
        DecisionsQuestionType.SCORE: ("instructions", "levels"),
        DecisionsQuestionType.PREDICATE: ("instructions",),
    }
    placeholder_fields: ClassVar[Mapping[str, Any]] = {
        "instructions": "placeholder",
        "choices": [{"value": "a"}, {"value": "b"}],
        "levels": [{"label": "a"}, {"label": "b"}],
    }
    generic_answer_schema: ClassVar[dict[str, Any]] = _GENERIC_ANSWER_SCHEMA

    def create_client(self, model: str) -> UiPathDecisionsClient:
        try:
            return UiPathDecisionsClient(
                model_name=model, timeout=DECISIONS_TIMEOUT_SECONDS
            )
        except Exception as exc:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                title="OpenAI Decisions is not available",
                detail=(
                    "Could not configure access to the OpenAI Decisions API "
                    f"through the LLM Gateway: {exc}"
                ),
                category=UiPathErrorCategory.SYSTEM,
            ) from exc

    async def ask(
        self,
        client: UiPathDecisionsClient,
        input_value: Any,
        questions: list[ClassifierQuestion],
        images: list[str],
    ) -> Any:
        return await client.acreate(
            build_decisions_input(input_value, images),
            build_decisions_questions(questions),
        )

    def _check_probabilities(
        self, question: DecisionsQuestion, probabilities: Any
    ) -> None:
        if not isinstance(probabilities, list) or not all(
            isinstance(item, dict) and is_finite_number(item.get("probability"))
            for item in probabilities
        ):
            raise self.missing_answer_error(
                question, f"malformed answer (probabilities={probabilities!r})"
            )

    def _check_answer(self, question: DecisionsQuestion, answer: Any) -> dict[str, Any]:
        """Return the answer to ``question`` without its name, once well formed.

        Only what the tool output promises is checked: the answer's type, its
        required fields and, for choice and score, its probabilities. Any other
        field the API returns is passed through.
        """
        if not isinstance(answer, dict):
            raise self.missing_answer_error(question, "no answer returned")
        if answer.get("type") != question.type.value:
            raise self.missing_answer_error(
                question,
                f"expected a {question.type.value} answer, got type "
                f"{answer.get('type')!r}",
            )
        for field in _ANSWER_FIELDS[question.type]:
            value = answer.get(field)
            valid = (
                isinstance(value, str) if field == "choice" else is_finite_number(value)
            )
            if not valid:
                raise self.missing_answer_error(
                    question, f"malformed answer ({field}={value!r})"
                )
        if not isinstance(question, DecisionsPredicateQuestion):
            self._check_probabilities(question, answer.get("probabilities"))
        return {key: value for key, value in answer.items() if key != "name"}

    def convert_answers(
        self, questions: list[ClassifierQuestion], response: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Return the answer to each question keyed by name.

        A question the model refused to answer is a tool error the agent sees: the
        refusal depends on the content, not on the tool configuration.
        """
        raw_answers = response.get("answers")
        if not isinstance(raw_answers, list):
            raise self.invalid_response_error(
                f"Expected a list of answers, got {raw_answers!r}."
            )
        answers = {
            answer.get("name"): answer
            for answer in raw_answers
            if isinstance(answer, dict)
        }
        refused = [
            question.name
            for question in questions
            if (answers.get(question.name) or {}).get("type") == "refusal"
        ]
        if refused:
            names = ", ".join(f"'{name}'" for name in refused)
            raise ToolException(
                f"{self.display_name} refused to answer the question(s) {names} "
                "about this input; no answer was given for any question."
            )
        return {
            question.name: self._check_answer(
                question,  # type: ignore[arg-type]
                answers.get(question.name),
            )
            for question in questions
        }

    def provider_message(self, error: UiPathAPIError) -> str:
        body = error.body
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict) and isinstance(inner.get("message"), str):
                return inner["message"]
            if inner is not None:
                return str(inner)
        return str(body) if body else error.message

    def is_unknown_model_error(self, error: UiPathAPIError) -> bool:
        """OpenAI answers ``{"error": {"code": "model_not_found", ...}}``."""
        if error.status_code not in {400, 404}:
            return False
        inner = error.body.get("error") if isinstance(error.body, dict) else None
        return isinstance(inner, dict) and inner.get("code") == "model_not_found"
