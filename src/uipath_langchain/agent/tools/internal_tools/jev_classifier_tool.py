"""Jev classifier internal tool.

Jev (TypeSafe AI) is a classification model, not a chat model: it answers typed
questions (``choice``, ``score``, ``noul``) about a state with calibrated
probabilities. The tool sends its ``state`` and ``questions`` arguments to Jev and
returns one answer object per question.
"""

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from httpx import HTTPError
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, TypeAdapter, ValidationError
from uipath.agent.models.agent import (
    JEV_NAME_PATTERN,
    AgentInternalJevClassifierToolProperties,
    AgentInternalToolResourceConfig,
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolObjectBuilderArgumentProperties,
    AgentToolStaticArgumentProperties,
    AgentToolTextBuilderArgumentProperties,
    JevChoiceQuestion,
    JevQuestion,
    JevQuestionType,
    JevScoreQuestion,
    TextTokenType,
)
from uipath.eval.mocks import mockable
from uipath.llm_client import UiPathAPIError
from uipath.llm_client.clients.typesafe import UiPathJevClient
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)
from uipath_langchain.agent.exceptions.llm import raise_for_provider_http_error
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model,
    create_output_model,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

JEV_STATE_ARGUMENT = "state"
JEV_QUESTIONS_ARGUMENT = "questions"

_STATE_PATH = f"$['{JEV_STATE_ARGUMENT}']"
_QUESTIONS_PATH = f"$['{JEV_QUESTIONS_ARGUMENT}']"
_STATE_TYPES: dict[str, type] = {"string": str, "object": dict, "array": list}
_QUESTION_TYPES = [question_type.value for question_type in JevQuestionType]
# Jev rejects an input it cannot take (invalid, or over its context length).
_INPUT_ERROR_STATUSES = frozenset({400, 413, 422})
# Jev answers in well under a second; a call still waiting after this is stuck.
JEV_TIMEOUT_SECONDS = 30.0
_INVALID_RESPONSE_TITLE = "Invalid Jev response"


class JevQuestionsMode(str, Enum):
    """Who supplies the questions, from the configuration at ``$['questions']``."""

    STATIC = "static"
    PROMPT = "prompt"
    ARGUMENT = "argument"
    BUILDER = "builder"


_CONFIDENCE_SCHEMA: dict[str, Any] = {
    "type": "number",
    "description": "Confidence in the answer, from 0 to 1.",
}


def _probabilities_schema(description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": {"type": "number"},
        "description": description,
    }


def _legend_schema(description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": description,
    }


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
        "confidence": _CONFIDENCE_SCHEMA,
        "probabilities": _probabilities_schema(
            "Probability of each option, keyed by option name (choice), or of each "
            "level, keyed by level index (score)."
        ),
        "legend": _legend_schema(
            "The text of each level, keyed by level index (score)."
        ),
        "noul": {
            "type": "number",
            "description": "Probability that the answer is yes, from 0 to 1 (noul).",
        },
    },
    "required": ["type"],
    "additionalProperties": True,
}

_QUESTION_ADAPTER: TypeAdapter[JevQuestion] = TypeAdapter(JevQuestion)

# Stand-ins for the fields of a built question that are not static, so a static
# field can be checked with the question models before any call.
_PLACEHOLDER_FIELDS: dict[str, Any] = {
    "instructions": "placeholder",
    "options": [{"name": "a"}, {"name": "b"}],
    "levels": ["a", "b"],
    "criteria": None,
}

_BUILDER_FIELDS: dict[JevQuestionType, tuple[str, ...]] = {
    JevQuestionType.CHOICE: ("instructions", "options"),
    JevQuestionType.SCORE: ("instructions", "levels"),
    # Optional: static {} (the default) or null means no criteria.
    JevQuestionType.NOUL: ("instructions", "criteria"),
}


def _question_path(name: str, field: str | None = None) -> str:
    path = f"{_QUESTIONS_PATH}['{name}']"
    return f"{path}['{field}']" if field else path


@dataclass(frozen=True)
class _JevConfig:
    """What the tool needs to know about its arguments, read once at startup."""

    mode: JevQuestionsMode
    state_type: str
    # Builder only: the fixed type of each question, by name.
    builder_types: dict[str, JevQuestionType]
    # The resource's argument properties plus, in builder mode, the pinned types.
    argument_properties: dict[str, AgentToolArgumentProperties]
    llm_writes_state: bool
    llm_writes_questions: bool


def _startup_error(tool_name: str, detail: str) -> AgentStartupError:
    return AgentStartupError(
        code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
        title="Invalid Jev classifier tool configuration",
        detail=f"Tool '{tool_name}': {detail}",
        category=UiPathErrorCategory.USER,
    )


def _questions_mode(
    tool_name: str, argument_properties: Mapping[str, AgentToolArgumentProperties]
) -> JevQuestionsMode:
    match argument_properties.get(_QUESTIONS_PATH):
        case None:
            return JevQuestionsMode.PROMPT
        case AgentToolStaticArgumentProperties():
            return JevQuestionsMode.STATIC
        case AgentToolArgumentArgumentProperties():
            return JevQuestionsMode.ARGUMENT
        case AgentToolObjectBuilderArgumentProperties():
            return JevQuestionsMode.BUILDER
        case other:
            raise _startup_error(
                tool_name,
                f"questions cannot be supplied by a {other.variant.value} "
                "(use a static value, the LLM, an agent input or the questions "
                "builder).",
            )


def _builder_types(
    tool_name: str, questions_schema: dict[str, Any]
) -> dict[str, JevQuestionType]:
    """Read the fixed type of each built question from its single-value enum."""
    properties = questions_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise _startup_error(
            tool_name, "inputSchema.properties.questions declares no questions."
        )
    types: dict[str, JevQuestionType] = {}
    for name, schema in properties.items():
        where = f"inputSchema.properties.questions.properties.{name}"
        if not re.fullmatch(JEV_NAME_PATTERN, name):
            raise _startup_error(
                tool_name,
                f"question name '{name}' must start with a letter or underscore "
                "and contain only letters, digits and underscores (at most 64).",
            )
        type_schema = (
            (schema.get("properties") or {}).get("type")
            if isinstance(schema, dict) and schema.get("type") == "object"
            else None
        )
        enum = type_schema.get("enum") if isinstance(type_schema, dict) else None
        if (
            not isinstance(type_schema, dict)
            or type_schema.get("type") != "string"
            or not isinstance(enum, list)
            or len(enum) != 1
            or enum[0] not in _QUESTION_TYPES
        ):
            raise _startup_error(
                tool_name,
                f"{where} must be an object whose 'type' property is a string "
                f"with a single-value enum, one of {', '.join(_QUESTION_TYPES)}.",
            )
        types[name] = JevQuestionType(enum[0])
    return types


def _llm_writes(
    argument_properties: Mapping[str, AgentToolArgumentProperties],
    path: str,
    schema: Any,
) -> bool:
    """Whether the LLM fills any part of the value at ``path``.

    An unconfigured value is the LLM's; an object builder is only a marker, so the
    answer comes from its declared properties.
    """
    props = argument_properties.get(path)
    if props is None:
        return True
    if not isinstance(props, AgentToolObjectBuilderArgumentProperties):
        return False
    properties = schema.get("properties") if isinstance(schema, dict) else None
    return any(
        _llm_writes(argument_properties, f"{path}['{name}']", sub_schema)
        for name, sub_schema in (properties or {}).items()
    )


def _read_config(
    tool_name: str,
    input_schema: dict[str, Any],
    argument_properties: Mapping[str, AgentToolArgumentProperties],
) -> _JevConfig:
    """Check ``inputSchema`` against the questions mode (AgentStartupError)."""
    properties = input_schema.get("properties") or {}
    required = input_schema.get("required") or []
    state_schema = properties.get(JEV_STATE_ARGUMENT)
    if not isinstance(state_schema, dict) or state_schema.get("type") not in (
        _STATE_TYPES
    ):
        raise _startup_error(
            tool_name,
            "inputSchema.properties.state must be a string, object or array schema.",
        )
    questions_schema = properties.get(JEV_QUESTIONS_ARGUMENT)
    if not isinstance(questions_schema, dict):
        raise _startup_error(tool_name, "inputSchema.properties.questions is missing.")
    missing = [
        name
        for name in (JEV_STATE_ARGUMENT, JEV_QUESTIONS_ARGUMENT)
        if name not in required
    ]
    if missing:
        raise _startup_error(
            tool_name,
            f"inputSchema.required must list {' and '.join(missing)}.",
        )

    mode = _questions_mode(tool_name, argument_properties)
    builder_types: dict[str, JevQuestionType] = {}
    if mode is JevQuestionsMode.BUILDER:
        if questions_schema.get("type") != "object":
            raise _startup_error(
                tool_name,
                "inputSchema.properties.questions must be an object keyed by "
                "question name when the questions are built one by one.",
            )
        builder_types = _builder_types(tool_name, questions_schema)
    elif questions_schema.get("type") != "array":
        raise _startup_error(
            tool_name,
            "inputSchema.properties.questions must be an array of questions.",
        )

    # The type of a built question is fixed by its schema: pin it like any static
    # value, so it is merged into the call and never left to the LLM.
    augmented = dict(argument_properties)
    for name, question_type in builder_types.items():
        augmented[_question_path(name, "type")] = AgentToolStaticArgumentProperties(
            value=question_type.value, is_sensitive=False
        )

    return _JevConfig(
        mode=mode,
        state_type=state_schema["type"],
        builder_types=builder_types,
        argument_properties=augmented,
        llm_writes_state=_llm_writes(augmented, _STATE_PATH, state_schema),
        llm_writes_questions=_llm_writes(augmented, _QUESTIONS_PATH, questions_schema),
    )


def _describe_validation_error(error: ValidationError) -> str:
    return "; ".join(
        (
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            if item["loc"]
            else item["msg"]
        )
        for item in error.errors()
    )


def parse_jev_questions(
    value: Any, builder_types: Mapping[str, JevQuestionType] | None = None
) -> list[JevQuestion]:
    """Validate the ``questions`` argument and return the questions it holds.

    ``value`` is a list of questions, or, when ``builder_types`` is given, an object
    keyed by question name whose types are the given ones.

    Raises:
        ValueError: listing every problem found, worded for whoever wrote them.
    """
    items: list[Any]
    if builder_types:
        if not isinstance(value, dict):
            raise ValueError("'questions' must be an object keyed by question name.")
        items = []
        for name, question_type in builder_types.items():
            fields = value.get(name)
            if fields is not None and not isinstance(fields, dict):
                raise ValueError(f"Question '{name}' must be an object.")
            items.append({**(fields or {}), "name": name, "type": question_type.value})
    else:
        if not isinstance(value, list) or not value:
            raise ValueError("'questions' must be a non-empty list of questions.")
        items = value

    errors: list[str] = []
    questions: list[JevQuestion] = []
    names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Question {index + 1} must be an object.")
            continue
        label = item.get("name") if isinstance(item.get("name"), str) else index + 1
        try:
            question = _QUESTION_ADAPTER.validate_python(item)
        except ValidationError as exc:
            errors.append(f"Question '{label}': {_describe_validation_error(exc)}")
            continue
        except ValueError as exc:
            errors.append(f"Question '{label}': {exc}")
            continue
        if question.name in names:
            errors.append(
                f"Question name '{question.name}' is used more than once; each "
                "question needs a unique name."
            )
            continue
        names.add(question.name)
        questions.append(question)
    if errors:
        raise ValueError("Invalid Jev questions:\n- " + "\n- ".join(errors))
    return questions


def _static_value(
    argument_properties: Mapping[str, AgentToolArgumentProperties], path: str
) -> Any:
    props = argument_properties.get(path)
    if isinstance(props, AgentToolStaticArgumentProperties):
        return props.value
    return None


def _fixed_text(
    argument_properties: Mapping[str, AgentToolArgumentProperties], path: str
) -> str | None:
    """The text at ``path`` when it is fixed at design time, else None.

    Agent Builder writes a static string as a text builder: it is fixed when all
    its tokens are plain text. A ``static`` string value is fixed too.
    """
    props = argument_properties.get(path)
    if isinstance(props, AgentToolStaticArgumentProperties):
        return props.value if isinstance(props.value, str) else None
    if isinstance(props, AgentToolTextBuilderArgumentProperties) and all(
        token.type == TextTokenType.SIMPLE_TEXT for token in props.tokens
    ):
        return "".join(token.raw_string for token in props.tokens)
    return None


def _builder_field_value(
    argument_properties: Mapping[str, AgentToolArgumentProperties],
    name: str,
    field: str,
) -> Any:
    """The fixed value of a builder question field, or a stand-in when not fixed."""
    path = _question_path(name, field)
    if field == "instructions":
        text = _fixed_text(argument_properties, path)
        return text if text is not None else _PLACEHOLDER_FIELDS[field]
    field_props = argument_properties.get(path)
    if isinstance(field_props, AgentToolStaticArgumentProperties):
        return field_props.value
    return _PLACEHOLDER_FIELDS[field]


def _check_builder_questions(config: _JevConfig) -> None:
    """Check each static builder field, with stand-ins for the other ones."""
    for name, question_type in config.builder_types.items():
        fields = {
            field: _builder_field_value(config.argument_properties, name, field)
            for field in _BUILDER_FIELDS[question_type]
        }
        parse_jev_questions({name: fields}, {name: question_type})


def _check_static_questions(tool_name: str, config: _JevConfig) -> None:
    """Validate the questions fixed by the configuration (AgentStartupError)."""
    try:
        if config.mode is JevQuestionsMode.STATIC:
            parse_jev_questions(
                _static_value(config.argument_properties, _QUESTIONS_PATH)
            )
        elif config.mode is JevQuestionsMode.BUILDER:
            _check_builder_questions(config)
    except ValueError as exc:
        raise _startup_error(
            tool_name, f"the static questions are invalid. {exc}"
        ) from exc


# The output when the resource declares none (Prompt, Argument, or an agent.json
# written without Agent Builder): answers keyed by question name.
_GENERIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Answers keyed by question name.",
    "properties": {},
    "additionalProperties": _GENERIC_ANSWER_SCHEMA,
}


def _output_schema(output_schema: dict[str, Any]) -> dict[str, Any]:
    """The resource's ``outputSchema``, or generic answers when it declares none.

    Agent Builder derives the ``outputSchema`` from the questions on every change.
    """
    if output_schema.get("properties") or "additionalProperties" in output_schema:
        return output_schema
    return _GENERIC_OUTPUT_SCHEMA


def _to_json(value: Any) -> Any:
    """Turn validated tool arguments back into plain JSON, dropping unset fields.

    Explicit values, nulls included, are kept so a value given whole (argument or
    static value) reaches Jev unchanged.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True, exclude_unset=True)
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json(item) for item in value]
    return value


def _is_state(value: Any, state_type: str) -> bool:
    """Whether ``value`` is a non-empty state of the declared type."""
    if not isinstance(value, _STATE_TYPES[state_type]):
        return False
    return bool(value.strip() if isinstance(value, str) else value)


def build_jev_questions(questions: list[JevQuestion]) -> dict[str, dict[str, Any]]:
    """Translate validated questions into TypeSafe's ``questions`` payload."""
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
        elif question.criteria is not None:
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


def _invalid_response_error(detail: str) -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
        title=_INVALID_RESPONSE_TITLE,
        detail=detail,
        category=UiPathErrorCategory.SYSTEM,
    )


def _missing_answer_error(question: JevQuestion, detail: str) -> AgentRuntimeError:
    return _invalid_response_error(f"Question '{question.name}': {detail}")


# The fields every answer of a type has, each a number except ``choice``.
_ANSWER_FIELDS: dict[JevQuestionType, tuple[str, ...]] = {
    JevQuestionType.CHOICE: ("choice", "confidence"),
    JevQuestionType.SCORE: ("score", "confidence"),
    JevQuestionType.NOUL: ("noul",),
}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_answer(question: JevQuestion, answer: Any) -> dict[str, Any]:
    """Return Jev's answer to ``question`` unchanged, once it is well formed.

    Only what the tool output promises is checked: the answer's type, its required
    fields and, for choice and score, its probabilities. Any other field Jev
    returns (the score ``legend``, fields added later) is passed through.
    """
    if not isinstance(answer, dict):
        raise _missing_answer_error(question, "no answer returned")
    if answer.get("type") != question.type.value:
        raise _missing_answer_error(
            question,
            f"expected a {question.type.value} answer, got type {answer.get('type')!r}",
        )
    for field in _ANSWER_FIELDS[question.type]:
        value = answer.get(field)
        valid = (
            isinstance(value, str) if field == "choice" else _is_finite_number(value)
        )
        if not valid:
            raise _missing_answer_error(
                question, f"malformed answer ({field}={value!r})"
            )
    if question.type is not JevQuestionType.NOUL:
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not all(
            _is_finite_number(value) for value in probabilities.values()
        ):
            raise _missing_answer_error(
                question, f"malformed answer (probabilities={probabilities!r})"
            )
    return answer


def convert_jev_answers(
    questions: list[JevQuestion], response: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return Jev's answer to each question, as Jev gave it, keyed by question name.

    Answers to questions that were not asked are left out.
    """
    answers = response.get("answers") or {}
    return {
        question.name: _check_answer(question, answers.get(question.name))
        for question in questions
    }


def _input_error(llm_wrote_it: bool, title: str, detail: str) -> Exception:
    """A tool error the LLM can correct, or a configuration error."""
    if llm_wrote_it:
        return ToolException(f"{detail}\nFix the arguments and call the tool again.")
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.INVALID_INPUT_ARGUMENT,
        title=title,
        detail=detail,
        category=UiPathErrorCategory.USER,
    )


def _provider_message(error: UiPathAPIError) -> str:
    body = error.body
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            return detail["message"]
        if detail is not None:
            return str(detail)
    return str(body) if body else error.message


def _is_unknown_model_error(error: UiPathAPIError) -> bool:
    """Whether Jev rejected the call because the configured model does not exist.

    TypeSafe answers 400 ``{"detail": {"message": "Unknown model: <name>"}}`` and the
    LLM Gateway 400 ``{"detail": {"message": "Unsupported model used. ..."}}``; a
    FastAPI-style 422 on ``body.model`` is accepted too. The model comes from the
    tool settings, so the LLM cannot fix it.
    """
    if error.status_code not in _INPUT_ERROR_STATUSES:
        return False
    detail = error.body.get("detail") if isinstance(error.body, dict) else None
    if isinstance(detail, list):
        return any(
            isinstance(item, dict) and list(item.get("loc") or [])[-1:] == ["model"]
            for item in detail
        )
    return (
        _provider_message(error)
        .lower()
        .startswith(("unknown model", "unsupported model"))
    )


def _read_tool_input(
    config: _JevConfig, kwargs: Mapping[str, Any]
) -> tuple[Any, list[JevQuestion]]:
    """The state and the parsed questions from the tool arguments."""
    state = _to_json(kwargs.get(JEV_STATE_ARGUMENT))
    if not _is_state(state, config.state_type):
        raise _input_error(
            config.llm_writes_state,
            "Missing state to classify",
            f"Argument '{JEV_STATE_ARGUMENT}' must be a non-empty {config.state_type}.",
        )
    try:
        questions = parse_jev_questions(
            _to_json(kwargs.get(JEV_QUESTIONS_ARGUMENT)), config.builder_types
        )
    except ValueError as exc:
        raise _input_error(
            config.llm_writes_questions, "Invalid Jev questions", str(exc)
        ) from exc
    return state, questions


def _create_jev_client(model: str) -> UiPathJevClient:
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


def _raise_for_jev_api_error(
    error: UiPathAPIError, model: str, resource_name: str, llm_writes_input: bool
) -> None:
    """Raise the tool error for a call Jev rejected."""
    if _is_unknown_model_error(error):
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST,
            title="Unknown Jev model",
            detail=(
                f"Jev has no model named '{model}'. Choose another "
                f"model in the settings of tool '{resource_name}'."
            ),
            category=UiPathErrorCategory.USER,
            status=error.status_code,
        ) from error
    if llm_writes_input and error.status_code in _INPUT_ERROR_STATUSES:
        raise ToolException(
            f"Jev rejected the input (HTTP {error.status_code}): "
            f"{_provider_message(error)}\nFix the arguments (the state and "
            "questions together must also fit Jev's context length) and "
            "call the tool again."
        ) from error
    raise_for_provider_http_error(error)


async def _ask_jev(
    client: UiPathJevClient,
    state: Any,
    questions: list[JevQuestion],
    *,
    model: str,
    resource_name: str,
    llm_writes_input: bool,
) -> dict[str, Any]:
    """Send the questions to Jev and return its response object."""
    try:
        response = await client.asystem_one(state, build_jev_questions(questions))
    except UiPathAPIError as exc:
        _raise_for_jev_api_error(exc, model, resource_name, llm_writes_input)
        raise
    except HTTPError as exc:
        # Timeouts and connection failures that outlived the client's retries.
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.HTTP_ERROR,
            title="Jev is not reachable",
            detail=f"The call to Jev failed: {type(exc).__name__}: {exc}",
            category=UiPathErrorCategory.SYSTEM,
        ) from exc
    except json.JSONDecodeError as exc:
        raise _invalid_response_error(
            f"Jev returned a response that is not JSON: {exc}"
        ) from exc
    if not isinstance(response, dict):
        raise _invalid_response_error(
            "Jev returned a response that is not a JSON object."
        )
    return response


class JevClassifierTool(StructuredToolWithArgumentProperties):
    """The Jev classifier tool.

    ``handle_tool_error`` turns a ``ToolException`` into an error tool result the
    LLM sees. Arguments that do not match the input schema become one too, when
    the LLM wrote any part of them: the base class turns pydantic's validation
    error into an ``AgentRuntimeError`` before LangChain's
    ``handle_validation_error`` could see it.
    """

    llm_writes_input: bool = False

    def _invalid_input_error(self, error: ValidationError) -> Exception:
        if self.llm_writes_input:
            return ToolException(
                f"Invalid arguments for tool '{self.name}'. Fix them to match the "
                f"tool input schema and call the tool again.\n\n{error}"
            )
        return super()._invalid_input_error(error)


def create_jev_classifier_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the jev-classifier internal tool from resource configuration.

    ``llm`` is accepted for signature parity with the other internal tool
    factories but is unused: classification is done by Jev.
    """
    properties = resource.properties
    if not isinstance(properties, AgentInternalJevClassifierToolProperties):
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Invalid Jev classifier tool configuration",
            detail=f"Expected Jev classifier tool properties for '{resource.name}'.",
            category=UiPathErrorCategory.USER,
        )
    settings = properties.settings
    config = _read_config(
        resource.name, resource.input_schema, resource.argument_properties
    )
    _check_static_questions(resource.name, config)

    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_output_model(
        _output_schema(resource.output_schema), resource.name
    )
    llm_writes_input = config.llm_writes_state or config.llm_writes_questions

    client: UiPathJevClient | None = None

    def get_client() -> UiPathJevClient:
        nonlocal client
        if client is None:
            client = _create_jev_client(settings.model)
        return client

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for internal tools
    )
    async def jev_classifier_tool_fn(**kwargs: Any) -> dict[str, Any]:
        state, questions = _read_tool_input(config, kwargs)
        response = await _ask_jev(
            get_client(),
            state,
            questions,
            model=settings.model,
            resource_name=resource.name,
            llm_writes_input=llm_writes_input,
        )
        return convert_jev_answers(questions, response)

    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    tool = JevClassifierTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=jev_classifier_tool_fn,
        output_type=output_model,
        argument_properties=config.argument_properties,
        llm_writes_input=llm_writes_input,
        handle_tool_error=True,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(awrapper=job_attachment_wrapper)
    return tool
