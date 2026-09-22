"""Jev classifier internal tool.

Jev (TypeSafe AI) is a classification model, not a chat model: it answers typed
questions (``choice``, ``score``, ``noul``) about an input text with calibrated
probabilities. The tool sends the ``text`` argument as Jev's ``state`` together
with the questions configured on the resource, and returns one answer object per
question.

Access goes through the LLM Gateway, unless ``TYPESAFE_API_KEY`` is set, in which
case TypeSafe is called directly (testing path until the gateway serves Jev).
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalJevClassifierSettings,
    AgentInternalJevClassifierToolProperties,
    AgentInternalToolResourceConfig,
    JevChoiceQuestion,
    JevQuestion,
    JevScoreQuestion,
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

JEV_TEXT_ARGUMENT = "text"

JEV_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        JEV_TEXT_ARGUMENT: {"type": "string", "description": "The text to classify."}
    },
    "required": [JEV_TEXT_ARGUMENT],
}

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


def _question_output_schema(question: JevQuestion) -> dict[str, Any]:
    if isinstance(question, JevChoiceQuestion):
        properties: dict[str, Any] = {
            "choice": {
                "type": "string",
                "enum": [option.name for option in question.options],
                "description": "The selected option.",
            },
            "confidence": _CONFIDENCE_SCHEMA,
            "probabilities": _probabilities_schema("Probability of each option."),
        }
    elif isinstance(question, JevScoreQuestion):
        legend = ", ".join(f"{i}={level}" for i, level in enumerate(question.levels))
        properties = {
            "score": {
                "type": "number",
                "description": (
                    "Probability-weighted score from 0 to "
                    f"{len(question.levels) - 1}; can land between levels. "
                    f"Levels: {legend}"
                ),
            },
            "level": {
                "type": "string",
                "enum": list(question.levels),
                "description": "The level closest to the score.",
            },
            "confidence": _CONFIDENCE_SCHEMA,
            "probabilities": _probabilities_schema(
                "Probability of each level, keyed by level index."
            ),
        }
    else:
        properties = {
            "noul": {
                "type": "number",
                "description": (
                    "Probability that the answer is yes, from 0 (no) to 1 (yes)."
                ),
            },
            "answer": {
                "type": "boolean",
                "description": "True when the probability is at least 0.5.",
            },
        }
    return {
        "type": "object",
        "description": question.instructions,
        "properties": properties,
        "required": list(properties),
    }


def build_jev_output_schema(
    settings: AgentInternalJevClassifierSettings,
) -> dict[str, Any]:
    """Build the tool output JSON schema: one answer object per question.

    Matches the schema Agent Builder writes to the resource's ``outputSchema``.
    """
    return {
        "type": "object",
        "properties": {
            question.name: _question_output_schema(question)
            for question in settings.questions
        },
        "required": [question.name for question in settings.questions],
    }


def build_jev_questions(
    settings: AgentInternalJevClassifierSettings,
) -> dict[str, dict[str, Any]]:
    """Translate the configured questions into TypeSafe ``questions`` payload."""
    questions: dict[str, dict[str, Any]] = {}
    for question in settings.questions:
        payload: dict[str, Any] = {
            "type": question.type.value,
            "instructions": question.instructions,
        }
        if isinstance(question, JevChoiceQuestion):
            payload["criteria"] = {
                option.name: option.description or None for option in question.options
            }
        elif isinstance(question, JevScoreQuestion):
            payload["criteria"] = list(question.levels)
        questions[question.name] = payload
    return questions


def _missing_answer_error(question: JevQuestion, detail: str) -> AgentRuntimeError:
    return AgentRuntimeError(
        code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
        title="Invalid Jev response",
        detail=f"Question '{question.name}': {detail}",
        category=UiPathErrorCategory.SYSTEM,
    )


def _convert_answer(question: JevQuestion, answer: Any) -> dict[str, Any]:
    if not isinstance(answer, dict):
        raise _missing_answer_error(question, "no answer returned")
    try:
        if isinstance(question, JevChoiceQuestion):
            return {
                "choice": str(answer["choice"]),
                "confidence": float(answer["confidence"]),
                "probabilities": dict(answer.get("probabilities") or {}),
            }
        if isinstance(question, JevScoreQuestion):
            score = float(answer["score"])
            index = min(max(round(score), 0), len(question.levels) - 1)
            return {
                "score": score,
                "level": question.levels[index],
                "confidence": float(answer["confidence"]),
                "probabilities": dict(answer.get("probabilities") or {}),
            }
        probability = float(answer["noul"])
        return {"noul": probability, "answer": probability >= 0.5}
    except (KeyError, TypeError, ValueError) as exc:
        raise _missing_answer_error(question, f"malformed answer ({exc!r})") from exc


def convert_jev_answers(
    settings: AgentInternalJevClassifierSettings, response: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Map a TypeSafe response onto the tool output schema."""
    answers = response.get("answers") or {}
    return {
        question.name: _convert_answer(question, answers.get(question.name))
        for question in settings.questions
    }


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
    jev_questions = build_jev_questions(settings)

    tool_name = sanitize_tool_name(resource.name)
    # The input is fixed; the stored schema is used only to keep a description the
    # user edited in Agent Builder.
    stored_properties = resource.input_schema.get("properties") or {}
    input_schema = (
        resource.input_schema
        if JEV_TEXT_ARGUMENT in stored_properties
        else JEV_INPUT_SCHEMA
    )
    input_model = create_model(input_schema)
    # Derived from settings rather than resource.output_schema, so the returned
    # answers always validate against the declared output.
    output_model = create_output_model(build_jev_output_schema(settings), resource.name)

    client: UiPathJevClient | None = None

    def get_client() -> UiPathJevClient:
        nonlocal client
        if client is None:
            try:
                client = UiPathJevClient(model_name=settings.model)
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
        return client

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for internal tools
    )
    async def jev_classifier_tool_fn(**kwargs: Any) -> dict[str, Any]:
        text = kwargs.get(JEV_TEXT_ARGUMENT)
        if not isinstance(text, str) or not text.strip():
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.INVALID_INPUT_ARGUMENT,
                title="Missing text to classify",
                detail=f"Argument '{JEV_TEXT_ARGUMENT}' must be a non-empty string.",
                category=UiPathErrorCategory.USER,
            )
        try:
            response = await get_client().asystem_one(text, jev_questions)
        except UiPathAPIError as exc:
            raise_for_provider_http_error(exc)
        return convert_jev_answers(settings, response)

    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=jev_classifier_tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(awrapper=job_attachment_wrapper)
    return tool
