"""Settings of the Jev generic tool and their mapping to the Jev API.

The settings shape matches the ``generic`` tool contract authored by Agent
Builder (``properties.settings`` of a ``subType: "jev"`` tool). Questions are a
list so the designer keeps their order; the Jev API expects them as a map keyed
by question name.
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

JEV_DEFAULT_MODEL = "jev-latest"

_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"

# Limits documented by the Jev API.
_MIN_CHOICE_OPTIONS = 2
_MAX_CHOICE_OPTIONS = 255
_MIN_SCORE_LEVELS = 2
_MAX_SCORE_LEVELS = 10

JEV_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "state": {
            "type": "string",
            "description": (
                "The content to classify. Include all context needed for the "
                "decision (e.g. the message, ticket or conversation excerpt)."
            ),
        }
    },
    "required": ["state"],
}

_PROBABILITIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "number"},
}


class _JevCfg(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class JevChoiceOption(_JevCfg):
    """One option of a choice question."""

    name: str = Field(..., pattern=_NAME_PATTERN)
    description: str | None = None


class JevChoiceQuestion(_JevCfg):
    """Pick exactly one option out of a fixed set."""

    type: Literal["choice"] = "choice"
    name: str = Field(..., pattern=_NAME_PATTERN)
    instructions: str = Field(..., min_length=1)
    options: list[JevChoiceOption] = Field(
        ..., min_length=_MIN_CHOICE_OPTIONS, max_length=_MAX_CHOICE_OPTIONS
    )

    @model_validator(mode="after")
    def _unique_options(self) -> "JevChoiceQuestion":
        names = [option.name for option in self.options]
        if len(names) != len(set(names)):
            raise ValueError(f"Question '{self.name}' has duplicate option names.")
        return self

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": {option.name: option.description for option in self.options},
        }

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": self.instructions,
            "properties": {
                "choice": {
                    "type": "string",
                    "enum": [option.name for option in self.options],
                },
                "confidence": {"type": "number"},
                "probabilities": _PROBABILITIES_SCHEMA,
            },
            "required": ["choice", "confidence", "probabilities"],
        }

    def to_output(self, answer: dict[str, Any]) -> dict[str, Any]:
        return {
            "choice": answer["choice"],
            "confidence": answer["confidence"],
            "probabilities": answer["probabilities"],
        }


class JevScoreQuestion(_JevCfg):
    """Rate on an ordered rubric; the score is a probability-weighted level index."""

    type: Literal["score"] = "score"
    name: str = Field(..., pattern=_NAME_PATTERN)
    instructions: str = Field(..., min_length=1)
    levels: list[Annotated[str, Field(min_length=1)]] = Field(
        ..., min_length=_MIN_SCORE_LEVELS, max_length=_MAX_SCORE_LEVELS
    )

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.levels),
        }

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": self.instructions,
            "properties": {
                "score": {"type": "number"},
                "confidence": {"type": "number"},
                "probabilities": _PROBABILITIES_SCHEMA,
            },
            "required": ["score", "confidence", "probabilities"],
        }

    def to_output(self, answer: dict[str, Any]) -> dict[str, Any]:
        return {
            "score": answer["score"],
            "confidence": answer["confidence"],
            "probabilities": answer["probabilities"],
        }


class JevNoulQuestion(_JevCfg):
    """Probability (0..1) that a statement is true."""

    type: Literal["noul"] = "noul"
    name: str = Field(..., pattern=_NAME_PATTERN)
    instructions: str = Field(..., min_length=1)
    true_description: str | None = Field(None, alias="trueDescription")
    false_description: str | None = Field(None, alias="falseDescription")

    def to_api(self) -> dict[str, Any]:
        question: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        criteria = {
            key: value
            for key, value in (
                ("true", self.true_description),
                ("false", self.false_description),
            )
            if value
        }
        if criteria:
            question["criteria"] = criteria
        return question

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": self.instructions,
            "properties": {"noul": {"type": "number"}},
            "required": ["noul"],
        }

    def to_output(self, answer: dict[str, Any]) -> dict[str, Any]:
        return {"noul": answer["noul"]}


JevQuestion = Annotated[
    Union[JevChoiceQuestion, JevScoreQuestion, JevNoulQuestion],
    Field(discriminator="type"),
]


class JevToolSettings(_JevCfg):
    """``properties.settings`` of a Jev generic tool."""

    model: str = Field(default=JEV_DEFAULT_MODEL, min_length=1)
    questions: list[JevQuestion] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_questions(self) -> "JevToolSettings":
        names = [question.name for question in self.questions]
        if len(names) != len(set(names)):
            raise ValueError("Question names must be unique.")
        return self

    def api_questions(self) -> dict[str, dict[str, Any]]:
        return {question.name: question.to_api() for question in self.questions}

    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                question.name: question.output_schema() for question in self.questions
            },
            "required": [question.name for question in self.questions],
        }

    def to_output(self, answers: dict[str, Any]) -> dict[str, Any]:
        """Map the API ``answers`` to the tool output.

        Raises:
            KeyError: If an answer or one of its fields is missing.
        """
        return {
            question.name: question.to_output(answers[question.name])
            for question in self.questions
        }
