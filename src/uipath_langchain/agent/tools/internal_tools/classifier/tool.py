"""Classifier internal tool.

A classifier answers typed questions about an input with probabilities, through a
classification model rather than a chat model. The model's provider sets the
shapes: TypeSafe's Jev classifies a ``state`` and answers ``choice``, ``score`` and
``noul`` questions; OpenAI's Decisions API classifies an ``input`` (text and
optional ``images``) and answers ``choice``, ``score`` and ``predicate`` questions.
The tool sends its arguments to the provider and returns one answer object per
question, keyed by question name.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from httpx import HTTPError
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ValidationError
from uipath.agent.models.agent import (
    CLASSIFIER_QUESTION_NAME_PATTERN,
    AgentInternalClassifierToolProperties,
    AgentInternalToolResourceConfig,
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    AgentToolObjectBuilderArgumentProperties,
    AgentToolStaticArgumentProperties,
    AgentToolTextBuilderArgumentProperties,
    ClassifierProvider,
    ClassifierQuestion,
    TextTokenType,
)
from uipath.eval.mocks import mockable
from uipath.llm_client import UiPathAPIError
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)
from uipath_langchain.agent.exceptions.llm import raise_for_provider_http_error
from uipath_langchain.agent.multimodal import (
    download_file_base64,
    is_image,
    normalize_mime_type,
)
from uipath_langchain.agent.multimodal.types import MAX_FILE_SIZE_BYTES
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model,
    create_output_model,
)
from uipath_langchain.agent.tools.internal_tools.analyze_files_tool import (
    resolve_attachments_to_file_infos,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

from .decisions import DecisionsProvider
from .provider import INPUT_ERROR_STATUSES, ClassifierProviderAdapter
from .typesafe import TypeSafeProvider

QUESTIONS_ARGUMENT = "questions"

_QUESTIONS_PATH = f"$['{QUESTIONS_ARGUMENT}']"
_INPUT_TYPES: dict[str, type] = {"string": str, "object": dict, "array": list}

PROVIDERS: dict[ClassifierProvider, ClassifierProviderAdapter] = {
    provider.provider: provider
    for provider in (TypeSafeProvider(), DecisionsProvider())
}


class QuestionsMode(str, Enum):
    """Who supplies the questions, from the configuration at ``$['questions']``."""

    STATIC = "static"
    PROMPT = "prompt"
    ARGUMENT = "argument"
    BUILDER = "builder"


def _argument_path(name: str) -> str:
    return f"$['{name}']"


def _question_path(name: str, field: str | None = None) -> str:
    path = f"{_QUESTIONS_PATH}['{name}']"
    return f"{path}['{field}']" if field else path


@dataclass(frozen=True)
class _ClassifierConfig:
    """What the tool needs to know about its arguments, read once at startup."""

    mode: QuestionsMode
    input_type: str
    # Builder only: the fixed type of each question, by name.
    builder_types: dict[str, Any]
    # The resource's argument properties plus, in builder mode, the pinned types.
    argument_properties: dict[str, AgentToolArgumentProperties]
    llm_writes_input: bool
    llm_writes_questions: bool
    llm_writes_images: bool

    @property
    def llm_writes_any(self) -> bool:
        return (
            self.llm_writes_input or self.llm_writes_questions or self.llm_writes_images
        )


def _startup_error(tool_name: str, detail: str) -> AgentStartupError:
    return AgentStartupError(
        code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
        title="Invalid classifier tool configuration",
        detail=f"Tool '{tool_name}': {detail}",
        category=UiPathErrorCategory.USER,
    )


def _questions_mode(
    tool_name: str, argument_properties: Mapping[str, AgentToolArgumentProperties]
) -> QuestionsMode:
    match argument_properties.get(_QUESTIONS_PATH):
        case None:
            return QuestionsMode.PROMPT
        case AgentToolStaticArgumentProperties():
            return QuestionsMode.STATIC
        case AgentToolArgumentArgumentProperties():
            return QuestionsMode.ARGUMENT
        case AgentToolObjectBuilderArgumentProperties():
            return QuestionsMode.BUILDER
        case other:
            raise _startup_error(
                tool_name,
                f"questions cannot be supplied by a {other.variant.value} "
                "(use a static value, the LLM, an agent input or the questions "
                "builder).",
            )


def _builder_types(
    tool_name: str,
    provider: ClassifierProviderAdapter,
    questions_schema: dict[str, Any],
) -> dict[str, Any]:
    """Read the fixed type of each built question from its single-value enum."""
    question_types = [question_type.value for question_type in provider.question_types]
    properties = questions_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise _startup_error(
            tool_name, "inputSchema.properties.questions declares no questions."
        )
    types: dict[str, Any] = {}
    for name, schema in properties.items():
        where = f"inputSchema.properties.questions.properties.{name}"
        if not re.fullmatch(CLASSIFIER_QUESTION_NAME_PATTERN, name):
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
            or enum[0] not in question_types
        ):
            raise _startup_error(
                tool_name,
                f"{where} must be an object whose 'type' property is a string "
                f"with a single-value enum, one of {', '.join(question_types)}.",
            )
        types[name] = provider.question_types(enum[0])
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
    provider: ClassifierProviderAdapter,
    input_schema: dict[str, Any],
    argument_properties: Mapping[str, AgentToolArgumentProperties],
) -> _ClassifierConfig:
    """Check ``inputSchema`` against the provider and questions mode."""
    input_argument = provider.input_argument
    properties = input_schema.get("properties") or {}
    required = input_schema.get("required") or []
    value_schema = properties.get(input_argument)
    if not isinstance(value_schema, dict) or value_schema.get("type") not in (
        _INPUT_TYPES
    ):
        raise _startup_error(
            tool_name,
            f"inputSchema.properties.{input_argument} must be a string, object or "
            "array schema.",
        )
    questions_schema = properties.get(QUESTIONS_ARGUMENT)
    if not isinstance(questions_schema, dict):
        raise _startup_error(tool_name, "inputSchema.properties.questions is missing.")
    missing = [
        name for name in (input_argument, QUESTIONS_ARGUMENT) if name not in required
    ]
    if missing:
        raise _startup_error(
            tool_name,
            f"inputSchema.required must list {' and '.join(missing)}.",
        )
    images_schema = (
        properties.get(provider.images_argument) if provider.images_argument else None
    )
    if images_schema is not None and (
        not isinstance(images_schema, dict) or images_schema.get("type") != "array"
    ):
        raise _startup_error(
            tool_name,
            f"inputSchema.properties.{provider.images_argument} must be an array "
            "of files.",
        )

    mode = _questions_mode(tool_name, argument_properties)
    builder_types: dict[str, Any] = {}
    if mode is QuestionsMode.BUILDER:
        if questions_schema.get("type") != "object":
            raise _startup_error(
                tool_name,
                "inputSchema.properties.questions must be an object keyed by "
                "question name when the questions are built one by one.",
            )
        builder_types = _builder_types(tool_name, provider, questions_schema)
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

    return _ClassifierConfig(
        mode=mode,
        input_type=value_schema["type"],
        builder_types=builder_types,
        argument_properties=augmented,
        llm_writes_input=_llm_writes(
            augmented, _argument_path(input_argument), value_schema
        ),
        llm_writes_questions=_llm_writes(augmented, _QUESTIONS_PATH, questions_schema),
        llm_writes_images=(
            images_schema is not None
            and provider.images_argument is not None
            and _llm_writes(
                augmented, _argument_path(provider.images_argument), images_schema
            )
        ),
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


def parse_classifier_questions(
    value: Any,
    provider: ClassifierProviderAdapter,
    builder_types: Mapping[str, Any] | None = None,
) -> list[ClassifierQuestion]:
    """Validate the ``questions`` argument and return the questions it holds.

    ``value`` is a list of questions in the provider's shapes, or, when
    ``builder_types`` is given, an object keyed by question name whose types are
    the given ones.

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
    questions: list[ClassifierQuestion] = []
    names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"Question {index + 1} must be an object.")
            continue
        label = item.get("name") if isinstance(item.get("name"), str) else index + 1
        try:
            question = provider.question_adapter.validate_python(item)
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
        raise ValueError(
            f"Invalid {provider.display_name} questions:\n- " + "\n- ".join(errors)
        )
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
    provider: ClassifierProviderAdapter,
    argument_properties: Mapping[str, AgentToolArgumentProperties],
    name: str,
    field: str,
) -> Any:
    """The fixed value of a builder question field, or a stand-in when not fixed."""
    path = _question_path(name, field)
    if field == "instructions":
        text = _fixed_text(argument_properties, path)
        return text if text is not None else provider.placeholder_fields[field]
    field_props = argument_properties.get(path)
    if isinstance(field_props, AgentToolStaticArgumentProperties):
        return field_props.value
    return provider.placeholder_fields[field]


def _check_builder_questions(
    provider: ClassifierProviderAdapter, config: _ClassifierConfig
) -> None:
    """Check each static builder field, with stand-ins for the other ones."""
    for name, question_type in config.builder_types.items():
        fields = {
            field: _builder_field_value(
                provider, config.argument_properties, name, field
            )
            for field in provider.builder_fields[question_type]
        }
        parse_classifier_questions({name: fields}, provider, {name: question_type})


def _check_static_questions(
    tool_name: str, provider: ClassifierProviderAdapter, config: _ClassifierConfig
) -> None:
    """Validate the questions fixed by the configuration (AgentStartupError)."""
    try:
        if config.mode is QuestionsMode.STATIC:
            parse_classifier_questions(
                _static_value(config.argument_properties, _QUESTIONS_PATH), provider
            )
        elif config.mode is QuestionsMode.BUILDER:
            _check_builder_questions(provider, config)
    except ValueError as exc:
        raise _startup_error(
            tool_name, f"the static questions are invalid. {exc}"
        ) from exc


def _output_schema(
    provider: ClassifierProviderAdapter, output_schema: dict[str, Any]
) -> dict[str, Any]:
    """The resource's ``outputSchema``, or generic answers when it declares none.

    Agent Builder derives the ``outputSchema`` from the questions on every change.
    The generic output is used for Prompt and Argument questions, or an agent.json
    written without Agent Builder: answers keyed by question name.
    """
    if output_schema.get("properties") or "additionalProperties" in output_schema:
        return output_schema
    return {
        "type": "object",
        "description": "Answers keyed by question name.",
        "properties": {},
        "additionalProperties": provider.generic_answer_schema,
    }


def _to_json(value: Any) -> Any:
    """Turn validated tool arguments back into plain JSON, dropping unset fields.

    Explicit values, nulls included, are kept so a value given whole (argument or
    static value) reaches the provider unchanged.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True, exclude_unset=True)
    if isinstance(value, dict):
        return {k: _to_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json(item) for item in value]
    return value


def _is_input(value: Any, input_type: str) -> bool:
    """Whether ``value`` is a non-empty input of the declared type."""
    if not isinstance(value, _INPUT_TYPES[input_type]):
        return False
    return bool(value.strip() if isinstance(value, str) else value)


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


async def _read_images(
    provider: ClassifierProviderAdapter, config: _ClassifierConfig, value: Any
) -> list[str]:
    """The images argument (job attachments) as data URLs."""
    if not value:
        return []
    images: list[str] = []
    for file in await resolve_attachments_to_file_infos(list(value)):
        mime_type = normalize_mime_type(file.mime_type, file.name)
        if not is_image(mime_type):
            raise _input_error(
                config.llm_writes_images,
                "Unsupported image",
                f"File '{file.name}' ({mime_type or 'unknown type'}) is not an "
                f"image {provider.display_name} can classify.",
            )
        try:
            data = await download_file_base64(file.url, max_size=MAX_FILE_SIZE_BYTES)
        except ValueError as exc:
            raise _input_error(
                config.llm_writes_images,
                "Image too large",
                f"File '{file.name}': {exc}",
            ) from exc
        images.append(f"data:{mime_type};base64,{data}")
    return images


async def _read_tool_input(
    provider: ClassifierProviderAdapter,
    config: _ClassifierConfig,
    kwargs: Mapping[str, Any],
) -> tuple[Any, list[ClassifierQuestion], list[str]]:
    """The input, the parsed questions and the images from the tool arguments."""
    input_argument = provider.input_argument
    images = (
        await _read_images(
            provider, config, _to_json(kwargs.get(provider.images_argument))
        )
        if provider.images_argument
        else []
    )
    input_value = _to_json(kwargs.get(input_argument))
    # Images can be classified on their own: the text input may then be empty.
    if not _is_input(input_value, config.input_type) and not (
        images and isinstance(input_value, _INPUT_TYPES[config.input_type])
    ):
        raise _input_error(
            config.llm_writes_input,
            "Missing input to classify",
            f"Argument '{input_argument}' must be a non-empty {config.input_type}.",
        )
    try:
        questions = parse_classifier_questions(
            _to_json(kwargs.get(QUESTIONS_ARGUMENT)), provider, config.builder_types
        )
    except ValueError as exc:
        raise _input_error(
            config.llm_writes_questions,
            f"Invalid {provider.display_name} questions",
            str(exc),
        ) from exc
    return input_value, questions, images


def _raise_for_api_error(
    provider: ClassifierProviderAdapter,
    error: UiPathAPIError,
    model: str,
    resource_name: str,
    llm_writes_input: bool,
) -> None:
    """Raise the tool error for a call the provider rejected."""
    name = provider.display_name
    if provider.is_unknown_model_error(error):
        # The model comes from the tool settings, so the LLM cannot fix it.
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST,
            title=f"Unknown {name} model",
            detail=(
                f"{name} has no model named '{model}'. Choose another "
                f"model in the settings of tool '{resource_name}'."
            ),
            category=UiPathErrorCategory.USER,
            status=error.status_code,
        ) from error
    if llm_writes_input and error.status_code in INPUT_ERROR_STATUSES:
        raise ToolException(
            f"{name} rejected the input (HTTP {error.status_code}): "
            f"{provider.provider_message(error)}\nFix the arguments (the input "
            f"and questions together must also fit {name}'s context length) and "
            "call the tool again."
        ) from error
    raise_for_provider_http_error(error)


async def _ask(
    provider: ClassifierProviderAdapter,
    client: Any,
    input_value: Any,
    questions: list[ClassifierQuestion],
    images: list[str],
    *,
    model: str,
    resource_name: str,
    llm_writes_input: bool,
) -> dict[str, Any]:
    """Send the questions to the provider and return its response object."""
    name = provider.display_name
    try:
        response = await provider.ask(client, input_value, questions, images)
    except UiPathAPIError as exc:
        _raise_for_api_error(provider, exc, model, resource_name, llm_writes_input)
        raise
    except HTTPError as exc:
        # Timeouts and connection failures that outlived the client's retries.
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.HTTP_ERROR,
            title=f"{name} is not reachable",
            detail=f"The call to {name} failed: {type(exc).__name__}: {exc}",
            category=UiPathErrorCategory.SYSTEM,
        ) from exc
    except json.JSONDecodeError as exc:
        raise provider.invalid_response_error(
            f"{name} returned a response that is not JSON: {exc}"
        ) from exc
    if not isinstance(response, dict):
        raise provider.invalid_response_error(
            f"{name} returned a response that is not a JSON object."
        )
    return response


class ClassifierTool(StructuredToolWithArgumentProperties):
    """The classifier tool.

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


def create_classifier_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the classifier internal tool from resource configuration.

    ``llm`` is accepted for signature parity with the other internal tool
    factories but is unused: classification is done by the configured model.
    """
    properties = resource.properties
    if not isinstance(properties, AgentInternalClassifierToolProperties):
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Invalid classifier tool configuration",
            detail=f"Expected classifier tool properties for '{resource.name}'.",
            category=UiPathErrorCategory.USER,
        )
    settings = properties.settings
    provider = PROVIDERS[settings.provider]
    config = _read_config(
        resource.name, provider, resource.input_schema, resource.argument_properties
    )
    _check_static_questions(resource.name, provider, config)

    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_output_model(
        _output_schema(provider, resource.output_schema), resource.name
    )
    llm_writes_input = config.llm_writes_any

    client: Any = None

    def get_client() -> Any:
        nonlocal client
        if client is None:
            client = provider.create_client(settings.model)
        return client

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for internal tools
    )
    async def classifier_tool_fn(**kwargs: Any) -> dict[str, Any]:
        input_value, questions, images = await _read_tool_input(
            provider, config, kwargs
        )
        response = await _ask(
            provider,
            get_client(),
            input_value,
            questions,
            images,
            model=settings.model,
            resource_name=resource.name,
            llm_writes_input=llm_writes_input,
        )
        return provider.convert_answers(questions, response)

    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    tool = ClassifierTool(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=classifier_tool_fn,
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
