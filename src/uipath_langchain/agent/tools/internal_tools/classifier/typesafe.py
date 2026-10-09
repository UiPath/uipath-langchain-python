"""TypeSafe's Jev models as a classifier provider.

Jev is a classification model, not a chat model: it answers typed questions
(``choice``, ``score``, ``noul``) about a ``state`` with calibrated probabilities.
"""

from typing import Any, ClassVar, Mapping, Sequence

from pydantic import TypeAdapter
from uipath.agent.models.agent import (
    ClassifierProvider,
    ClassifierQuestion,
    JevChoiceQuestion,
    JevNoulQuestion,
    JevQuestion,
    JevQuestionType,
    JevScoreQuestion,
)
from uipath.llm_client import UiPathAPIError
from uipath.llm_client.clients.typesafe import UiPathJevClient
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

from .provider import (
    CONFIDENCE_SCHEMA,
    INPUT_ERROR_STATUSES,
    ClassifierProviderAdapter,
    is_finite_number,
)

# Jev answers in well under a second; a call still waiting after this is stuck.
JEV_TIMEOUT_SECONDS = 30.0

_QUESTION_TYPES = [question_type.value for question_type in JevQuestionType]

# One answer of the generic output (Prompt, Argument): the fields Jev returns for
# every type. Answers are Jev's as they are, so fields Jev adds later pass too.
_GENERIC_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": list(_QUESTION_TYPES),
            "description": "The question type.",
        },
        "choice": {"type": "string", "description": "The selected option (choice)."},
        "score": {
            "type": "number",
            "description": (
                "Probability-weighted level index, from 0 (lowest level) up; can "
                "land between levels (score)."
            ),
        },
        "confidence": CONFIDENCE_SCHEMA,
        "probabilities": {
            "type": "object",
            "additionalProperties": {"type": "number"},
            "description": (
                "Probability of each option, keyed by option name (choice), or of "
                "each level, keyed by level index (score)."
            ),
        },
        "legend": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "The text of each level, keyed by level index (score).",
        },
        "noul": {
            "type": "number",
            "description": "Probability that the answer is yes, from 0 to 1 (noul).",
        },
    },
    "required": ["type"],
    "additionalProperties": True,
}

# The fields every answer of a type has, each a number except ``choice``.
_ANSWER_FIELDS: dict[JevQuestionType, tuple[str, ...]] = {
    JevQuestionType.CHOICE: ("choice", "confidence"),
    JevQuestionType.SCORE: ("score", "confidence"),
    JevQuestionType.NOUL: ("noul",),
}


def build_jev_questions(
    questions: Sequence[ClassifierQuestion],
) -> dict[str, dict[str, Any]]:
    """Translate validated Jev questions into TypeSafe's ``questions`` payload."""
    payload: dict[str, dict[str, Any]] = {}
    for question in questions:
        entry: dict[str, Any] = {
            "type": question.type.value,
            "instructions": question.instructions,
        }
        if isinstance(question, JevChoiceQuestion):
            entry["criteria"] = {
                option.name: option.description or None for option in question.options
            }
        elif isinstance(question, JevScoreQuestion):
            entry["criteria"] = list(question.levels)
        elif isinstance(question, JevNoulQuestion) and question.criteria is not None:
            # Only the sides with text; no criteria at all when neither has any.
            sides = {
                key: text
                for key, text in (
                    ("true", question.criteria.true),
                    ("false", question.criteria.false),
                )
                if text and text.strip()
            }
            if sides:
                entry["criteria"] = sides
        payload[question.name] = entry
    return payload


class TypeSafeProvider(ClassifierProviderAdapter):
    """Jev, through the LLM Gateway's TypeSafe passthrough."""

    provider: ClassVar[ClassifierProvider] = ClassifierProvider.TYPESAFE
    display_name: ClassVar[str] = "Jev"
    input_argument: ClassVar[str] = "state"
    question_types: ClassVar[type[JevQuestionType]] = JevQuestionType
    question_adapter: ClassVar[TypeAdapter[Any]] = TypeAdapter(JevQuestion)
    builder_fields: ClassVar[Mapping[Any, tuple[str, ...]]] = {
        JevQuestionType.CHOICE: ("instructions", "options"),
        JevQuestionType.SCORE: ("instructions", "levels"),
        # Optional: static {} (the default) or null means no criteria.
        JevQuestionType.NOUL: ("instructions", "criteria"),
    }
    placeholder_fields: ClassVar[Mapping[str, Any]] = {
        "instructions": "placeholder",
        "options": [{"name": "a"}, {"name": "b"}],
        "levels": ["a", "b"],
        "criteria": None,
    }
    generic_answer_schema: ClassVar[dict[str, Any]] = _GENERIC_ANSWER_SCHEMA

    def create_client(self, model: str) -> UiPathJevClient:
        try:
            return UiPathJevClient(model_name=model, timeout=JEV_TIMEOUT_SECONDS)
        except Exception as exc:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
                title="Jev is not available",
                detail=(
                    "Could not configure access to Jev through the LLM Gateway "
                    f"or with TYPESAFE_API_KEY: {exc}"
                ),
                category=UiPathErrorCategory.SYSTEM,
            ) from exc

    async def ask(
        self,
        client: UiPathJevClient,
        input_value: Any,
        questions: list[ClassifierQuestion],
        images: list[str],
    ) -> Any:
        return await client.asystem_one(
            input_value,
            build_jev_questions(questions),
        )

    def _check_answer(self, question: JevQuestion, answer: Any) -> dict[str, Any]:
        """Return Jev's answer to ``question`` unchanged, once it is well formed.

        Only what the tool output promises is checked: the answer's type, its
        required fields and, for choice and score, its probabilities. Any other
        field Jev returns (the score ``legend``, fields added later) is passed
        through.
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
        if not isinstance(question, JevNoulQuestion):
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or not all(
                is_finite_number(value) for value in probabilities.values()
            ):
                raise self.missing_answer_error(
                    question, f"malformed answer (probabilities={probabilities!r})"
                )
        return answer

    def convert_answers(
        self, questions: list[ClassifierQuestion], response: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Return Jev's answer to each question, as Jev gave it.

        Answers to questions that were not asked are left out.
        """
        answers = response.get("answers") or {}
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
            detail = body.get("detail")
            if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                return detail["message"]
            if detail is not None:
                return str(detail)
        return str(body) if body else error.message

    def is_unknown_model_error(self, error: UiPathAPIError) -> bool:
        """TypeSafe answers 400 ``{"detail": {"message": "Unknown model: <name>"}}``.

        A FastAPI-style 422 on ``body.model`` is accepted too.
        """
        if error.status_code not in INPUT_ERROR_STATUSES:
            return False
        detail = error.body.get("detail") if isinstance(error.body, dict) else None
        if isinstance(detail, list):
            return any(
                isinstance(item, dict) and list(item.get("loc") or [])[-1:] == ["model"]
                for item in detail
            )
        return self.provider_message(error).lower().startswith("unknown model")
