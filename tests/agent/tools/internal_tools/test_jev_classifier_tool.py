"""Tests for jev_classifier_tool.py module."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.tool import tool_call
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel
from uipath.agent.models.agent import AgentInternalToolResourceConfig
from uipath.llm_client import UiPathAPIError
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)
from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.internal_tools.internal_tool_factory import (
    create_internal_tool,
)
from uipath_langchain.agent.tools.internal_tools.jev_classifier_tool import (
    build_jev_questions,
    parse_jev_questions,
)
from uipath_langchain.agent.tools.static_args import StaticArgsHandler
from uipath_langchain.agent.tools.tool_node import create_tool_node

MODULE = "uipath_langchain.agent.tools.internal_tools.jev_classifier_tool"

pytestmark = pytest.mark.usefixtures("_passthrough_mockable")

# --- Schemas as Agent Builder writes them to the resource ----------------------

QUESTIONS_LIST_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "description": "The questions Jev answers about the state.",
    "items": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,63}$",
                "description": "Unique identifier of the question and the key of "
                "its answer in the tool output.",
            },
            "type": {
                "type": "string",
                "enum": ["choice", "score", "noul"],
                "description": "choice: pick one option. score: rate on ordered "
                "levels. noul: yes/no probability.",
            },
            "instructions": {
                "type": "string",
                "description": "The question to answer about the state, written "
                "as an instruction.",
            },
            "options": {
                "type": "array",
                "minItems": 2,
                "maxItems": 255,
                "description": "Required when type is choice; omit otherwise.",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The option's value, returned as the "
                            "answer.",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the option means.",
                        },
                    },
                },
            },
            "levels": {
                "type": "array",
                "minItems": 2,
                "maxItems": 10,
                "items": {"type": "string"},
                "description": "Required when type is score, from lowest to "
                "highest; omit otherwise.",
            },
            "criteria": {
                "type": "object",
                "description": "What counts as a yes or a no answer (noul); "
                "omit otherwise.",
                "properties": {
                    "true": {
                        "type": "string",
                        "description": "What counts as a yes answer.",
                    },
                    "false": {
                        "type": "string",
                        "description": "What counts as a no answer.",
                    },
                },
            },
        },
        "required": ["name", "type", "instructions"],
    },
}

_ITEM_PROPERTIES: dict[str, Any] = QUESTIONS_LIST_SCHEMA["items"]["properties"]
OPTION_ITEMS_SCHEMA: dict[str, Any] = _ITEM_PROPERTIES["options"]["items"]
INSTRUCTIONS_SCHEMA: dict[str, Any] = _ITEM_PROPERTIES["instructions"]
CRITERIA_SCHEMA: dict[str, Any] = {
    **_ITEM_PROPERTIES["criteria"],
    "description": "What counts as a yes or a no answer.",
}

BUILDER_QUESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["department", "frustration", "is_urgent"],
    "properties": {
        "department": {
            "type": "object",
            "required": ["type", "instructions", "options"],
            "properties": {
                "type": {"type": "string", "enum": ["choice"]},
                "instructions": INSTRUCTIONS_SCHEMA,
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 255,
                    "description": "Required when type is choice; omit otherwise.",
                    "items": OPTION_ITEMS_SCHEMA,
                },
            },
        },
        "frustration": {
            "type": "object",
            "required": ["type", "instructions", "levels"],
            "properties": {
                "type": {"type": "string", "enum": ["score"]},
                "instructions": INSTRUCTIONS_SCHEMA,
                "levels": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 10,
                    "items": {"type": "string"},
                    "description": "Required when type is score, from lowest to "
                    "highest; omit otherwise.",
                },
            },
        },
        "is_urgent": {
            "type": "object",
            "required": ["type", "instructions"],
            "properties": {
                "type": {"type": "string", "enum": ["noul"]},
                "instructions": INSTRUCTIONS_SCHEMA,
                "criteria": CRITERIA_SCHEMA,
            },
        },
    },
}

CONFIDENCE: dict[str, Any] = {
    "type": "number",
    "description": "Confidence in the answer, from 0 to 1.",
}

OPTION_PROBABILITIES: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "number"},
    "description": "Probability of each option, keyed by option name.",
}

LEVEL_PROBABILITIES: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "number"},
    "description": "Probability of each level, keyed by level index.",
}

LEGEND: dict[str, Any] = {
    "type": "object",
    "additionalProperties": {"type": "string"},
    "description": "The text of each level, keyed by level index, from lowest (0) "
    "to highest.",
}

NOUL_ANSWER_PROPERTIES: dict[str, Any] = {
    "type": {"type": "string", "enum": ["noul"]},
    "noul": {
        "type": "number",
        "description": "Probability that the answer is yes, from 0 (no) to 1 (yes).",
    },
}

CHOICE_REQUIRED = ["type", "choice", "confidence", "probabilities"]
SCORE_REQUIRED = ["type", "score", "confidence", "probabilities"]

STATIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "department": {
            "type": "object",
            "description": "Which team should handle this",
            "properties": {
                "type": {"type": "string", "enum": ["choice"]},
                "choice": {
                    "type": "string",
                    "enum": ["billing", "technical"],
                    "description": "The selected option.",
                },
                "confidence": CONFIDENCE,
                "probabilities": OPTION_PROBABILITIES,
            },
            "required": CHOICE_REQUIRED,
            "additionalProperties": True,
        },
        "frustration": {
            "type": "object",
            "description": "How frustrated the customer appears",
            "properties": {
                "type": {"type": "string", "enum": ["score"]},
                "score": {
                    "type": "number",
                    "description": "Probability-weighted level index from 0 to 2; "
                    "can land between levels. Levels: 0=Calm, 1=Frustrated, "
                    "2=Very angry",
                },
                "confidence": CONFIDENCE,
                "probabilities": LEVEL_PROBABILITIES,
                "legend": LEGEND,
            },
            "required": SCORE_REQUIRED,
            "additionalProperties": True,
        },
        "is_urgent": {
            "type": "object",
            "description": "Is this urgent?",
            "properties": NOUL_ANSWER_PROPERTIES,
            "required": ["type", "noul"],
            "additionalProperties": True,
        },
    },
    "required": ["department", "frustration", "is_urgent"],
}

GENERIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Answers keyed by question name.",
    "properties": {},
    "additionalProperties": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["choice", "score", "noul"],
                "description": "The question type.",
            },
            "choice": {
                "type": "string",
                "description": "The selected option (choice).",
            },
            "score": {
                "type": "number",
                "description": "Probability-weighted level index, from 0 (lowest "
                "level) up; can land between levels (score).",
            },
            "confidence": CONFIDENCE,
            "probabilities": {
                "type": "object",
                "additionalProperties": {"type": "number"},
                "description": "Probability of each option, keyed by option name "
                "(choice), or of each level, keyed by level index (score).",
            },
            "legend": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "The text of each level, keyed by level index (score).",
            },
            "noul": {
                "type": "number",
                "description": "Probability that the answer is yes, from 0 to 1 "
                "(noul).",
            },
        },
        "required": ["type"],
        "additionalProperties": True,
    },
}

# --- Fixtures -----------------------------------------------------------------

DEPARTMENT: dict[str, Any] = {
    "name": "department",
    "type": "choice",
    "instructions": "Which team should handle this",
    "options": [
        {"name": "billing", "description": "Payment issues"},
        {"name": "technical", "description": None},
    ],
}
FRUSTRATION: dict[str, Any] = {
    "name": "frustration",
    "type": "score",
    "instructions": "How frustrated the customer appears",
    "levels": ["Calm", "Frustrated", "Very angry"],
}
IS_URGENT: dict[str, Any] = {
    "name": "is_urgent",
    "type": "noul",
    "instructions": "Is this urgent?",
}
QUESTIONS: list[dict[str, Any]] = [DEPARTMENT, FRUSTRATION, IS_URGENT]

JEV_QUESTIONS_PAYLOAD: dict[str, Any] = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this",
        "criteria": {"billing": "Payment issues", "technical": None},
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated the customer appears",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
    "is_urgent": {"type": "noul", "instructions": "Is this urgent?"},
}

JEV_RESPONSE: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.81,
            "probabilities": {"billing": 0.88, "technical": 0.12},
        },
        "frustration": {
            "type": "score",
            "score": 1.6,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.0, "1": 0.4, "2": 0.6},
            "confidence": 0.7,
        },
        "is_urgent": {"type": "noul", "noul": 0.3},
    },
    "usage": {"input_tokens": 20, "output_tokens": 0},
}

# The tool output: Jev's answers, as Jev gave them.
ANSWERS: dict[str, Any] = JEV_RESPONSE["answers"]

STRING_STATE: dict[str, Any] = {"type": "string", "description": "Ticket body"}

OBJECT_STATE: dict[str, Any] = {
    "type": "object",
    "description": "The ticket",
    "properties": {
        "message": {"type": "string", "description": "Ticket body"},
        "order_id": {"type": "string"},
        "amount_usd": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "customer": {"type": "string"},
    },
    "required": ["message"],
}

MESSAGES_STATE: dict[str, Any] = {
    "type": "array",
    "description": "The conversation",
    "items": {
        "type": "object",
        "properties": {
            "from": {"type": "string", "description": "Who wrote it"},
            "text": {"type": "string"},
        },
        "required": ["from", "text"],
    },
}


def _static(value: Any) -> dict[str, Any]:
    return {"variant": "static", "value": value, "isSensitive": False}


def _argument(path: str) -> dict[str, Any]:
    return {"variant": "argument", "argumentPath": path, "isSensitive": False}


STATIC_QUESTIONS: dict[str, Any] = {"$['questions']": _static(QUESTIONS)}

BUILDER_ARGUMENT_PROPERTIES: dict[str, Any] = {
    "$['questions']": {"variant": "objectBuilder"},
    "$['questions']['department']": {"variant": "objectBuilder"},
    "$['questions']['department']['options']": _argument("$['departments']"),
    "$['questions']['frustration']": {"variant": "objectBuilder"},
    "$['questions']['frustration']['instructions']": {
        "variant": "textBuilder",
        "tokens": [
            {"type": "simpleText", "rawString": "How frustrated is "},
            {"type": "variable", "rawString": "input.customerName"},
        ],
        "isSensitive": False,
    },
    "$['questions']['frustration']['levels']": _static(FRUSTRATION["levels"]),
    "$['questions']['is_urgent']": {"variant": "objectBuilder"},
    # Agent Builder writes a fixed string as a text builder of plain text.
    "$['questions']['is_urgent']['instructions']": {
        "variant": "textBuilder",
        "tokens": [
            {"type": "simpleText", "rawString": "Does the customer need "},
            {"type": "simpleText", "rawString": "an answer today?"},
        ],
        "isSensitive": False,
    },
    # The editor's default for a noul question: no criteria.
    "$['questions']['is_urgent']['criteria']": _static({}),
}


def _input_schema(
    state: dict[str, Any] | None = None, questions: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "state": state if state is not None else STRING_STATE,
            "questions": questions if questions is not None else QUESTIONS_LIST_SCHEMA,
        },
        "required": ["state", "questions"],
    }


def _resource(
    argument_properties: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    questions: dict[str, Any] | None = None,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
) -> AgentInternalToolResourceConfig:
    return AgentInternalToolResourceConfig.model_validate(
        {
            "$resourceType": "tool",
            "name": "Classify Ticket",
            "description": "Classify a support ticket",
            "type": "Internal",
            "inputSchema": input_schema or _input_schema(state, questions),
            "outputSchema": output_schema or {"type": "object", "properties": {}},
            "properties": {
                "toolType": "jev-classifier",
                "settings": {"model": "jev-latest"},
            },
            "argumentProperties": (
                STATIC_QUESTIONS if argument_properties is None else argument_properties
            ),
        }
    )


def _builder_resource() -> AgentInternalToolResourceConfig:
    return _resource(BUILDER_ARGUMENT_PROPERTIES, questions=BUILDER_QUESTIONS_SCHEMA)


@pytest.fixture
def _passthrough_mockable():
    with patch(f"{MODULE}.mockable", lambda **kwargs: lambda f: f):
        yield


@pytest.fixture
def jev_client() -> Any:
    client = MagicMock()
    client.asystem_one = AsyncMock(return_value=JEV_RESPONSE)
    with patch(f"{MODULE}.UiPathJevClient", return_value=client) as client_cls:
        client.cls = client_cls
        yield client


class TicketInput(BaseModel):
    ticket: str = "I was charged twice!"
    customerName: str = "Ana"
    orderId: str = "A-104"
    departments: list[dict[str, Any]] = DEPARTMENT["options"]
    questionList: list[dict[str, Any]] = QUESTIONS


def _llm_tool(tool: BaseTool, agent_input: BaseModel | None = None) -> BaseTool:
    handler = StaticArgsHandler()
    agent_input = agent_input or TicketInput()
    [llm_tool] = handler.initialize([tool], agent_input, type(agent_input))
    return llm_tool


def _merged_call(
    tool: BaseTool, llm_args: dict[str, Any], agent_input: BaseModel | None = None
) -> ToolCall:
    """The tool call as the tool node receives it: LLM args plus configured ones."""
    handler = StaticArgsHandler()
    agent_input = agent_input or TicketInput()
    handler.initialize([tool], agent_input, type(agent_input))
    call = tool_call(name=tool.name, args=llm_args, id="call-1")
    handler.apply_to_response([call])
    return call


def _llm_schema(tool: BaseTool, *path: str) -> dict[str, Any]:
    """The LLM-facing schema of a (nested) argument."""
    schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
    node = schema
    for key in path:
        node = node["properties"][key]
        if "anyOf" in node:  # optional: T | None
            node = next(n for n in node["anyOf"] if n.get("type") != "null")
        if "$ref" in node:
            node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    return node


async def _tool_message(tool: BaseTool, args: dict[str, Any]) -> ToolMessage:
    result = await tool.ainvoke(tool_call(name=tool.name, args=args, id="call-1"))
    assert isinstance(result, ToolMessage)
    return result


# --- Tool schemas ---------------------------------------------------------------


def test_factory_uses_the_resource_schemas(jev_client: Any) -> None:
    tool = create_internal_tool(
        _resource(output_schema=STATIC_OUTPUT_SCHEMA), AsyncMock()
    )

    assert tool.name == "Classify_Ticket"
    assert tool.description == "Classify a support ticket"
    input_schema = tool.args_schema.model_json_schema()  # type: ignore[union-attr]
    assert list(input_schema["properties"]) == ["state", "questions"]
    assert input_schema["properties"]["state"]["description"] == "Ticket body"
    assert tool.metadata is not None
    output_schema = tool.metadata["output_schema"].model_json_schema()
    assert set(output_schema["properties"]) == {
        "department",
        "frustration",
        "is_urgent",
    }
    assert output_schema["required"] == ["department", "frustration", "is_urgent"]


def test_output_schema_without_answers_falls_back_to_generic(
    jev_client: Any,
) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())

    assert tool.metadata is not None
    schema = tool.metadata["output_schema"].model_json_schema()
    assert schema.get("properties", {}) == {}
    assert schema["description"] == "Answers keyed by question name."
    assert "$ref" in schema["additionalProperties"]


def test_generic_output_schema_of_the_resource_is_kept(jev_client: Any) -> None:
    declared = {**GENERIC_OUTPUT_SCHEMA, "description": "Answers by question"}
    tool = create_internal_tool(_resource({}, output_schema=declared), AsyncMock())

    assert tool.metadata is not None
    schema = tool.metadata["output_schema"].model_json_schema()
    assert schema["description"] == "Answers by question"


@pytest.mark.parametrize(
    ("criteria", "expected"),
    [
        pytest.param(
            {"true": "Needs a reply today", "false": "Can wait"},
            {"true": "Needs a reply today", "false": "Can wait"},
            id="both",
        ),
        pytest.param(
            {"true": "Needs a reply today", "false": ""},
            {"true": "Needs a reply today"},
            id="true-only",
        ),
        pytest.param({"false": "Can wait"}, {"false": "Can wait"}, id="false-only"),
        pytest.param({"true": " ", "false": None}, None, id="neither"),
        pytest.param({}, None, id="empty"),
        pytest.param(None, None, id="null"),
    ],
)
def test_noul_criteria_payload(
    criteria: dict[str, Any] | None, expected: dict[str, str] | None
) -> None:
    question = {**IS_URGENT, "criteria": criteria}

    [entry] = build_jev_questions(parse_jev_questions([question])).values()

    assert entry.get("criteria") == expected
    assert build_jev_questions(parse_jev_questions([IS_URGENT])) == {
        "is_urgent": {"type": "noul", "instructions": "Is this urgent?"}
    }


async def test_noul_criteria_from_the_llm_reach_jev(jev_client: Any) -> None:
    tool = create_internal_tool(_resource({}), AsyncMock())
    question = {**IS_URGENT, "criteria": {"true": "Needs a reply today"}}

    await tool.ainvoke({"state": "text", "questions": [question]})

    _, questions = jev_client.asystem_one.await_args.args
    assert questions == {
        "is_urgent": {
            "type": "noul",
            "instructions": "Is this urgent?",
            "criteria": {"true": "Needs a reply today"},
        }
    }


async def test_builder_criteria_merge_prompt_and_text_builder_sides(
    jev_client: Any,
) -> None:
    tool = create_internal_tool(
        _resource(
            {
                **BUILDER_ARGUMENT_PROPERTIES,
                "$['questions']['is_urgent']['criteria']": {"variant": "objectBuilder"},
                "$['questions']['is_urgent']['criteria']['false']": {
                    "variant": "textBuilder",
                    "tokens": [{"type": "simpleText", "rawString": "Can wait"}],
                    "isSensitive": False,
                },
            },
            questions=BUILDER_QUESTIONS_SCHEMA,
        ),
        AsyncMock(),
    )
    llm_tool = _llm_tool(tool)
    # 'true' is left to the LLM; 'false' is pinned.
    assert "enum" not in _llm_schema(
        llm_tool, "questions", "is_urgent", "criteria", "true"
    )
    assert _llm_schema(llm_tool, "questions", "is_urgent", "criteria", "false")[
        "enum"
    ] == ["Can wait"]

    call = _merged_call(
        tool,
        {
            "state": "text",
            "questions": {
                "department": {"instructions": "Which team"},
                "is_urgent": {"criteria": {"true": "Needs a reply today"}},
            },
        },
    )
    await tool.ainvoke(call["args"])

    _, questions = jev_client.asystem_one.await_args.args
    assert questions["is_urgent"] == {
        "type": "noul",
        "instructions": "Does the customer need an answer today?",
        "criteria": {"true": "Needs a reply today", "false": "Can wait"},
    }


# --- Startup validation -------------------------------------------------------


def _without(schema: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        **schema,
        "properties": {k: v for k, v in schema["properties"].items() if k != key},
    }


BUILDER_PROPS: dict[str, Any] = {"$['questions']": {"variant": "objectBuilder"}}


def _builder_entry(type_schema: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "q": {
                "type": "object",
                "properties": {"type": type_schema, "instructions": {"type": "string"}},
            }
        },
    }


@pytest.mark.parametrize(
    ("input_schema", "argument_properties", "detail"),
    [
        pytest.param(
            _without(_input_schema(), "state"), STATIC_QUESTIONS, "state", id="no-state"
        ),
        pytest.param(
            _input_schema(state={"type": "number"}),
            STATIC_QUESTIONS,
            "state must be a string, object or array",
            id="number-state",
        ),
        pytest.param(
            _without(_input_schema(), "questions"),
            STATIC_QUESTIONS,
            "questions is missing",
            id="no-questions",
        ),
        pytest.param(
            {**_input_schema(), "required": ["state"]},
            STATIC_QUESTIONS,
            "required must list questions",
            id="questions-not-required",
        ),
        pytest.param(
            _input_schema(questions={"type": "object"}),
            {},
            "must be an array of questions",
            id="prompt-with-object",
        ),
        pytest.param(
            _input_schema(questions={"type": "object"}),
            {"$['questions']": _argument("$['questionList']")},
            "must be an array of questions",
            id="argument-with-object",
        ),
        pytest.param(
            _input_schema(questions=QUESTIONS_LIST_SCHEMA),
            BUILDER_PROPS,
            "must be an object keyed by question name",
            id="builder-with-array",
        ),
        pytest.param(
            _input_schema(questions={"type": "object", "properties": {}}),
            BUILDER_PROPS,
            "declares no questions",
            id="builder-without-questions",
        ),
        pytest.param(
            _input_schema(questions=_builder_entry({"type": "string"})),
            BUILDER_PROPS,
            "single-value enum",
            id="builder-type-without-enum",
        ),
        pytest.param(
            _input_schema(
                questions=_builder_entry({"type": "string", "enum": ["choice", "noul"]})
            ),
            BUILDER_PROPS,
            "single-value enum",
            id="builder-type-with-two-values",
        ),
        pytest.param(
            _input_schema(
                questions=_builder_entry({"type": "string", "enum": ["freeform"]})
            ),
            BUILDER_PROPS,
            "single-value enum",
            id="builder-unknown-type",
        ),
        pytest.param(
            _input_schema(
                questions={
                    "type": "object",
                    "properties": {
                        "q\n": _builder_entry({"type": "string", "enum": ["noul"]})[
                            "properties"
                        ]["q"]
                    },
                }
            ),
            BUILDER_PROPS,
            "must start with a letter or underscore",
            id="builder-name-with-trailing-newline",
        ),
        pytest.param(
            _input_schema(
                questions={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                }
            ),
            BUILDER_PROPS,
            "single-value enum",
            id="builder-question-not-object",
        ),
        pytest.param(
            _input_schema(
                questions={
                    "type": "object",
                    "properties": {
                        "has space": _builder_entry(
                            {"type": "string", "enum": ["noul"]}
                        )["properties"]["q"]
                    },
                }
            ),
            BUILDER_PROPS,
            "question name 'has space'",
            id="builder-invalid-name",
        ),
        pytest.param(
            _input_schema(),
            {"$['questions']": {"variant": "arrayBuilder"}},
            "arrayBuilder",
            id="array-builder-questions",
        ),
        pytest.param(
            _input_schema(),
            {"$['questions']": _static([IS_URGENT, IS_URGENT])},
            "used more than once",
            id="static-duplicate-names",
        ),
        pytest.param(
            _input_schema(),
            {"$['questions']": _static([{**IS_URGENT, "levels": ["a", "b"]}])},
            "cannot have levels",
            id="static-levels-on-noul",
        ),
        pytest.param(
            _input_schema(),
            {"$['questions']": _static([{**DEPARTMENT, "options": [{"name": "a"}]}])},
            "options",
            id="static-one-option",
        ),
        pytest.param(
            _input_schema(),
            {"$['questions']": _static([])},
            "non-empty list",
            id="static-empty-list",
        ),
        pytest.param(
            _input_schema(questions=BUILDER_QUESTIONS_SCHEMA),
            {
                **BUILDER_ARGUMENT_PROPERTIES,
                "$['questions']['frustration']['levels']": _static(["only"]),
            },
            "frustration",
            id="builder-static-field-invalid",
        ),
        pytest.param(
            _input_schema(),
            {"$['questions']": _static([{**IS_URGENT, "criteria": {"true": 1}}])},
            "criteria",
            id="static-criteria-not-string",
        ),
        pytest.param(
            _input_schema(questions=BUILDER_QUESTIONS_SCHEMA),
            {
                **BUILDER_ARGUMENT_PROPERTIES,
                "$['questions']['is_urgent']['criteria']": _static("Yes"),
            },
            "criteria",
            id="builder-static-criteria-not-object",
        ),
        pytest.param(
            _input_schema(questions=BUILDER_QUESTIONS_SCHEMA),
            {
                **BUILDER_ARGUMENT_PROPERTIES,
                "$['questions']['is_urgent']['criteria']": _static({"yes": "x"}),
            },
            "yes",
            id="builder-static-criteria-unknown-key",
        ),
        pytest.param(
            _input_schema(questions=BUILDER_QUESTIONS_SCHEMA),
            {
                **BUILDER_ARGUMENT_PROPERTIES,
                "$['questions']['is_urgent']['instructions']": {
                    "variant": "textBuilder",
                    "tokens": [],
                    "isSensitive": False,
                },
            },
            "is_urgent",
            id="builder-empty-fixed-instructions",
        ),
    ],
)
def test_invalid_configuration_fails_at_startup(
    input_schema: dict[str, Any], argument_properties: dict[str, Any], detail: str
) -> None:
    resource = _resource(argument_properties, input_schema=input_schema)

    with pytest.raises(AgentStartupError) as exc_info:
        create_internal_tool(resource, AsyncMock())

    error = exc_info.value.error_info
    assert error.code == AgentStartupError.full_code(
        AgentStartupErrorCode.INVALID_TOOL_CONFIG
    )
    assert error.category == UiPathErrorCategory.USER
    assert detail in error.detail


# --- Questions modes, end to end ----------------------------------------------


async def test_static_questions(jev_client: Any) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())

    # Static questions are pinned (and visible) in the LLM-facing schema.
    assert _llm_schema(_llm_tool(tool), "questions")["enum"] == [json.dumps(QUESTIONS)]
    call = _merged_call(tool, {"state": "I was charged twice!", "questions": "x"})
    result = await tool.ainvoke(call["args"])

    jev_client.cls.assert_called_once_with(model_name="jev-latest", timeout=30.0)
    state, questions = jev_client.asystem_one.await_args.args
    assert state == "I was charged twice!"
    assert questions == JEV_QUESTIONS_PAYLOAD
    assert result == ANSWERS


async def test_prompt_questions(jev_client: Any) -> None:
    tool = create_internal_tool(_resource({}), AsyncMock())

    assert _llm_schema(_llm_tool(tool), "questions")["type"] == "array"
    result = await tool.ainvoke(
        {"state": "I was charged twice!", "questions": QUESTIONS}
    )

    _, questions = jev_client.asystem_one.await_args.args
    assert questions == JEV_QUESTIONS_PAYLOAD
    assert result == ANSWERS


async def test_argument_questions_resolve_levels_of_the_call(jev_client: Any) -> None:
    severity_answer = {
        "type": "score",
        "score": 0.8,
        "confidence": 0.9,
        "legend": {"0": "Low", "1": "High"},
        "probabilities": {"0": 0.2, "1": 0.8},
    }
    jev_client.asystem_one.return_value = {"answers": {"severity": severity_answer}}
    tool = create_internal_tool(
        _resource({"$['questions']": _argument("$['questionList']")}), AsyncMock()
    )
    severity = {
        "name": "severity",
        "type": "score",
        "instructions": "How severe",
        "levels": ["Low", "High"],
    }

    call = _merged_call(tool, {"state": "Down"}, TicketInput(questionList=[severity]))
    result = await tool.ainvoke(call["args"])

    _, questions = jev_client.asystem_one.await_args.args
    assert questions == {
        "severity": {
            "type": "score",
            "instructions": "How severe",
            "criteria": ["Low", "High"],
        }
    }
    assert result == {"severity": severity_answer}


async def test_builder_questions_merge_prompt_argument_and_static_fields(
    jev_client: Any,
) -> None:
    tool = create_internal_tool(_builder_resource(), AsyncMock())
    llm_tool = _llm_tool(tool)

    # Types are pinned from the schema; configured fields are pinned too; the LLM
    # is asked only for department's instructions.
    assert _llm_schema(llm_tool, "questions", "department", "type")["enum"] == [
        "choice"
    ]
    assert _llm_schema(llm_tool, "questions", "is_urgent", "type")["enum"] == ["noul"]
    assert "enum" not in _llm_schema(
        llm_tool, "questions", "department", "instructions"
    )
    assert _llm_schema(llm_tool, "questions", "department", "options")["enum"] == [
        json.dumps(DEPARTMENT["options"])
    ]
    assert _llm_schema(llm_tool, "questions", "is_urgent", "instructions")["enum"] == [
        "Does the customer need an answer today?"
    ]

    call = _merged_call(
        tool,
        {
            "state": "I was charged twice!",
            "questions": {
                "department": {"instructions": "Which team should handle this"}
            },
        },
    )
    result = await tool.ainvoke(call["args"])

    _, questions = jev_client.asystem_one.await_args.args
    assert questions == {
        "department": JEV_QUESTIONS_PAYLOAD["department"],
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is Ana",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does the customer need an answer today?",
        },
    }
    assert result == ANSWERS


async def test_builder_type_from_the_llm_is_overwritten(jev_client: Any) -> None:
    tool = create_internal_tool(_builder_resource(), AsyncMock())

    call = _merged_call(
        tool,
        {
            "state": "x",
            "questions": {"department": {"type": "noul", "instructions": "Which"}},
        },
    )

    assert call["args"]["questions"]["department"]["type"] == "choice"
    assert call["args"]["questions"]["is_urgent"]["type"] == "noul"


# --- Errors the LLM can fix ---------------------------------------------------


@pytest.mark.parametrize(
    ("questions", "detail"),
    [
        pytest.param([IS_URGENT, IS_URGENT], "used more than once", id="duplicate"),
        pytest.param(
            [{**IS_URGENT, "options": DEPARTMENT["options"]}],
            "cannot have options",
            id="options-on-noul",
        ),
        pytest.param(
            [{**DEPARTMENT, "levels": ["a", "b"]}],
            "cannot have levels",
            id="levels-on-choice",
        ),
        pytest.param(
            [{k: v for k, v in FRUSTRATION.items() if k != "levels"}],
            "levels",
            id="score-without-levels",
        ),
        pytest.param(
            [{**IS_URGENT, "instructions": ""}], "instructions", id="empty-instructions"
        ),
        pytest.param(
            [{**FRUSTRATION, "levels": ["Low", ""]}], "levels", id="empty-level"
        ),
        pytest.param(
            [{**DEPARTMENT, "options": [{"name": "a"}, {"name": "a"}]}],
            "duplicate option names",
            id="duplicate-options",
        ),
        pytest.param(
            [{**DEPARTMENT, "criteria": {"true": "Billing"}}],
            "cannot have criteria",
            id="criteria-on-choice",
        ),
        pytest.param(
            [{**IS_URGENT, "criteria": {"maybe": "Unclear"}}],
            "maybe",
            id="criteria-unknown-side",
        ),
        # These break the input schema itself (pattern, minItems, types).
        pytest.param(
            [{**IS_URGENT, "criteria": {"true": 1}}], "criteria", id="criteria-number"
        ),
        pytest.param([{**IS_URGENT, "name": "1st"}], "name", id="invalid-name"),
        pytest.param(
            [{**DEPARTMENT, "options": [{"name": "a"}]}], "options", id="one-option"
        ),
        pytest.param([], "questions", id="no-questions"),
    ],
)
async def test_invalid_llm_questions_are_a_tool_error(
    jev_client: Any, questions: list[dict[str, Any]], detail: str
) -> None:
    tool = create_internal_tool(_resource({}), AsyncMock())

    message = await _tool_message(tool, {"state": "text", "questions": questions})

    assert message.status == "error"
    assert detail in message.content
    assert "call the tool again" in message.content
    jev_client.asystem_one.assert_not_awaited()


async def test_invalid_builder_prompt_field_is_a_tool_error(jev_client: Any) -> None:
    tool = create_internal_tool(_builder_resource(), AsyncMock())
    call = _merged_call(
        tool, {"state": "x", "questions": {"department": {"instructions": ""}}}
    )

    message = await _tool_message(tool, call["args"])

    assert message.status == "error"
    assert "department" in message.content
    jev_client.asystem_one.assert_not_awaited()


async def test_empty_llm_state_is_a_tool_error(jev_client: Any) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())
    call = _merged_call(tool, {"state": "  "})

    message = await _tool_message(tool, call["args"])

    assert message.status == "error"
    assert "non-empty string" in message.content


@pytest.mark.parametrize("status", [400, 413, 422])
async def test_jev_input_error_for_llm_content_is_a_tool_error(
    jev_client: Any, status: int
) -> None:
    response = httpx.Response(
        status,
        json={"detail": {"message": "Context length exceeded"}},
        request=httpx.Request("POST", "https://api.typesafe.ai/v1/systemone"),
    )
    jev_client.asystem_one.side_effect = UiPathAPIError.from_response(response)
    tool = create_internal_tool(_resource({}), AsyncMock())

    message = await _tool_message(tool, {"state": "text", "questions": QUESTIONS})

    assert message.status == "error"
    assert "Context length exceeded" in message.content
    assert f"HTTP {status}" in message.content


# --- Configuration errors (not the LLM's to fix) --------------------------------

CONFIGURED_STATE: dict[str, Any] = {"$['state']": _argument("$['ticket']")}


async def test_invalid_argument_questions_are_a_runtime_error(jev_client: Any) -> None:
    tool = create_internal_tool(
        _resource(
            {**CONFIGURED_STATE, "$['questions']": _argument("$['questionList']")}
        ),
        AsyncMock(),
    )
    call = _merged_call(tool, {}, TicketInput(questionList=[IS_URGENT, IS_URGENT]))

    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(call)

    assert exc_info.value.error_info.category == UiPathErrorCategory.USER
    assert "used more than once" in exc_info.value.error_info.detail
    jev_client.asystem_one.assert_not_awaited()


async def test_argument_questions_breaking_the_schema_are_a_runtime_error(
    jev_client: Any,
) -> None:
    tool = create_internal_tool(
        _resource(
            {**CONFIGURED_STATE, "$['questions']": _argument("$['questionList']")}
        ),
        AsyncMock(),
    )
    call = _merged_call(
        tool, {}, TicketInput(questionList=[{**IS_URGENT, "name": "1st"}])
    )

    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(call)

    assert exc_info.value.error_info.category == UiPathErrorCategory.USER


async def test_invalid_argument_field_of_a_builder_is_a_runtime_error(
    jev_client: Any,
) -> None:
    tool = create_internal_tool(
        _resource(
            {
                **CONFIGURED_STATE,
                **BUILDER_ARGUMENT_PROPERTIES,
                "$['questions']['department']['instructions']": _static("Which"),
            },
            questions=BUILDER_QUESTIONS_SCHEMA,
        ),
        AsyncMock(),
    )
    duplicate = [{"name": "a"}, {"name": "a"}]
    call = _merged_call(tool, {}, TicketInput(departments=duplicate))

    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(call)

    assert "duplicate option names" in exc_info.value.error_info.detail


async def test_jev_input_error_without_llm_content_is_a_runtime_error(
    jev_client: Any,
) -> None:
    response = httpx.Response(
        422, request=httpx.Request("POST", "https://api.typesafe.ai/v1/systemone")
    )
    jev_client.asystem_one.side_effect = UiPathAPIError.from_response(response)
    tool = create_internal_tool(
        _resource({**CONFIGURED_STATE, **STATIC_QUESTIONS}), AsyncMock()
    )

    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(_merged_call(tool, {}))

    error = exc_info.value.error_info
    assert error.code == AgentRuntimeError.full_code(AgentRuntimeErrorCode.HTTP_ERROR)
    assert error.status == 422


async def test_jev_rate_limit_is_a_runtime_error(jev_client: Any) -> None:
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.typesafe.ai/v1/systemone")
    )
    jev_client.asystem_one.side_effect = UiPathAPIError.from_response(response)
    tool = create_internal_tool(_resource({}), AsyncMock())

    # A rate limit is not the LLM's to fix, even when the LLM wrote the input.
    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(
            tool_call(
                name=tool.name,
                args={"state": "text", "questions": QUESTIONS},
                id="call-1",
            )
        )

    error = exc_info.value.error_info
    assert error.code == AgentRuntimeError.full_code(AgentRuntimeErrorCode.HTTP_ERROR)
    assert error.status == 429


async def test_empty_configured_state_is_a_runtime_error(jev_client: Any) -> None:
    tool = create_internal_tool(
        _resource({**CONFIGURED_STATE, **STATIC_QUESTIONS}), AsyncMock()
    )

    with pytest.raises(AgentRuntimeError, match="non-empty"):
        await tool.ainvoke(_merged_call(tool, {}, TicketInput(ticket=" ")))


async def test_missing_answer_raises_runtime_error(jev_client: Any) -> None:
    jev_client.asystem_one.return_value = {"answers": {}}
    tool = create_internal_tool(_resource(), AsyncMock())

    with pytest.raises(AgentRuntimeError, match="department"):
        await tool.ainvoke(_merged_call(tool, {"state": "text"})["args"])


async def test_client_is_created_once(jev_client: Any) -> None:
    tool = create_internal_tool(_resource(), AsyncMock())

    await tool.ainvoke(_merged_call(tool, {"state": "one"})["args"])
    await tool.ainvoke(_merged_call(tool, {"state": "two"})["args"])

    jev_client.cls.assert_called_once()
    assert jev_client.asystem_one.await_count == 2


def _jev_error(status: int, body: Any) -> UiPathAPIError:
    response = httpx.Response(
        status,
        json=body,
        request=httpx.Request("POST", "https://api.typesafe.ai/v1/systemone"),
    )
    return UiPathAPIError.from_response(response)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(
            400,
            {
                "detail": {
                    "error_type": "api_usage_error",
                    "message": "Unknown model: jev-nonexistent",
                }
            },
            id="typesafe-400",
        ),
        pytest.param(
            422,
            {"detail": [{"loc": ["body", "model"], "msg": "Unknown model"}]},
            id="fastapi-422",
        ),
    ],
)
async def test_unknown_model_is_a_user_runtime_error_even_for_llm_input(
    jev_client: Any, status: int, body: Any
) -> None:
    jev_client.asystem_one.side_effect = _jev_error(status, body)
    tool = create_internal_tool(_resource({}), AsyncMock())

    with pytest.raises(AgentRuntimeError) as exc_info:
        await tool.ainvoke(
            tool_call(
                name=tool.name,
                args={"state": "text", "questions": QUESTIONS},
                id="call-1",
            )
        )

    error = exc_info.value.error_info
    assert error.category == UiPathErrorCategory.USER
    assert error.title == "Unknown Jev model"
    assert "jev-latest" in error.detail


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(httpx.ReadTimeout("timed out"), id="timeout"),
        pytest.param(httpx.ConnectError("refused"), id="connect-error"),
    ],
)
async def test_unreachable_jev_is_a_system_runtime_error(
    jev_client: Any, error: Exception
) -> None:
    jev_client.asystem_one.side_effect = error
    tool = create_internal_tool(_resource(), AsyncMock())

    with pytest.raises(AgentRuntimeError, match="The call to Jev failed") as exc_info:
        await tool.ainvoke(_merged_call(tool, {"state": "text"})["args"])

    assert exc_info.value.error_info.title == "Jev is not reachable"
    assert exc_info.value.error_info.category == UiPathErrorCategory.SYSTEM


async def test_non_json_response_is_an_invalid_response_error(jev_client: Any) -> None:
    jev_client.asystem_one.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    tool = create_internal_tool(_resource(), AsyncMock())

    with pytest.raises(AgentRuntimeError, match="not JSON"):
        await tool.ainvoke(_merged_call(tool, {"state": "text"})["args"])


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        pytest.param(
            "frustration",
            {"type": "score", "score": float("inf"), "confidence": 1.0},
            id="infinite-score",
        ),
        pytest.param(
            "frustration",
            {"type": "score", "score": float("nan"), "confidence": 1.0},
            id="nan-score",
        ),
        pytest.param(
            "is_urgent", {"type": "noul", "noul": float("nan")}, id="nan-noul"
        ),
        pytest.param("is_urgent", {"type": "noul", "noul": True}, id="boolean-noul"),
        pytest.param(
            "department",
            {"type": "choice", "choice": "billing", "confidence": 0.9},
            id="choice-without-probabilities",
        ),
        pytest.param(
            "frustration",
            {
                "type": "score",
                "score": 1.0,
                "confidence": 0.9,
                "probabilities": {"0": float("inf")},
            },
            id="non-finite-probability",
        ),
    ],
)
async def test_malformed_answer_is_an_invalid_response_error(
    jev_client: Any, question: str, answer: dict[str, Any]
) -> None:
    jev_client.asystem_one.return_value = {
        "answers": {**JEV_RESPONSE["answers"], question: answer}
    }
    tool = create_internal_tool(_resource(), AsyncMock())

    with pytest.raises(AgentRuntimeError, match="malformed answer") as exc_info:
        await tool.ainvoke(_merged_call(tool, {"state": "text"})["args"])

    assert question in exc_info.value.error_info.detail


async def test_answer_of_another_type_is_an_invalid_response_error(
    jev_client: Any,
) -> None:
    jev_client.asystem_one.return_value = {
        "answers": {
            **JEV_RESPONSE["answers"],
            "is_urgent": JEV_RESPONSE["answers"]["department"],
        }
    }
    tool = create_internal_tool(_resource(), AsyncMock())

    with pytest.raises(AgentRuntimeError, match="expected a noul answer"):
        await tool.ainvoke(_merged_call(tool, {"state": "text"})["args"])


async def test_answers_are_passed_through_with_fields_jev_adds(
    jev_client: Any,
) -> None:
    answers = {
        name: {**answer, "rationale_id": f"r-{name}"}
        for name, answer in JEV_RESPONSE["answers"].items()
    }
    jev_client.asystem_one.return_value = {
        **JEV_RESPONSE,
        "answers": {**answers, "not_asked": {"type": "noul", "noul": 1.0}},
    }
    tool = create_internal_tool(_resource(), AsyncMock())

    result = await tool.ainvoke(_merged_call(tool, {"state": "text"})["args"])

    assert result == answers


async def test_client_configuration_failure_is_runtime_error() -> None:
    with patch(f"{MODULE}.UiPathJevClient", side_effect=ValueError("no settings")):
        tool = create_internal_tool(_resource(), AsyncMock())

        with pytest.raises(AgentRuntimeError, match="configure access to Jev"):
            await tool.ainvoke(_merged_call(tool, {"state": "text"})["args"])


# --- Tool errors in the agent graph -------------------------------------------


async def test_tool_error_passes_through_the_tool_node_wrappers(
    jev_client: Any,
) -> None:
    """The node the autonomous agent runs: job-attachment wrapper, no error wrapper."""
    tool = create_internal_tool(_resource({}), AsyncMock())
    [node] = create_tool_node([tool]).values()
    call = tool_call(
        name=tool.name,
        args={"state": "text", "questions": [IS_URGENT, IS_URGENT]},
        id="call-1",
    )
    state = AgentGraphState(messages=[AIMessage(content="", tool_calls=[call])])

    command = await node.ainvoke(state)

    assert isinstance(command, Command)
    assert isinstance(command.update, dict)
    [message] = command.update["messages"]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert message.tool_call_id == "call-1"
    assert "used more than once" in message.content


class _ScriptedModel(GenericFakeChatModel):
    """Replays scripted messages and ignores the bound tools."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> BaseChatModel:
        return self


class _Output(BaseModel):
    department: str = ""


async def test_autonomous_agent_gets_invalid_questions_back_and_continues(
    jev_client: Any,
) -> None:
    tool = create_internal_tool(_resource({}), AsyncMock())
    model = _ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            name=tool.name,
                            args={"state": "text", "questions": [IS_URGENT, IS_URGENT]},
                            id="call-1",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            name=tool.name,
                            args={"state": "text", "questions": QUESTIONS},
                            id="call-2",
                        )
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        tool_call(
                            name="end_execution",
                            args={"department": "billing"},
                            id="call-3",
                        )
                    ],
                ),
            ]
        )
    )
    graph: Any = create_agent(
        model=model,
        tools=[tool],
        messages=[SystemMessage(content="sys"), HumanMessage(content="go")],
        output_schema=_Output,
    ).compile(checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "jev"}}

    result = await graph.ainvoke({}, config)

    assert result == {"department": "billing"}
    messages = (await graph.aget_state(config)).values["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.status) for m in tool_messages] == [
        ("call-1", "error"),
        ("call-2", "success"),
    ]
    assert "used more than once" in tool_messages[0].content
    jev_client.asystem_one.assert_awaited_once()


# --- State shapes, read from inputSchema --------------------------------------


@pytest.mark.parametrize(
    ("state_schema", "state"),
    [
        pytest.param(STRING_STATE, "I was charged twice!", id="string"),
        pytest.param(
            OBJECT_STATE,
            {"message": "Charged twice", "amount_usd": 49, "tags": ["refund"]},
            id="declared-object",
        ),
        pytest.param(
            {"type": "object", "description": "The ticket"},
            {"id": "A-104", "lines": [1, "two", {"three": 3}]},
            id="undeclared-object",
        ),
        pytest.param(
            MESSAGES_STATE,
            [
                {"from": "customer", "text": "I was charged twice."},
                {"from": "support", "text": "Checking."},
            ],
            id="list",
        ),
    ],
)
async def test_state_shapes_reach_jev_as_json(
    jev_client: Any, state_schema: dict[str, Any], state: Any
) -> None:
    tool = create_internal_tool(_resource(state=state_schema), AsyncMock())

    llm_state = _llm_schema(_llm_tool(tool), "state")
    await tool.ainvoke(_merged_call(tool, {"state": state})["args"])

    assert llm_state["type"] == state_schema["type"]
    assert llm_state.get("description") == state_schema["description"]
    sent, _ = jev_client.asystem_one.await_args.args
    assert sent == state


@pytest.mark.parametrize(
    ("state_schema", "state"),
    [
        pytest.param(STRING_STATE, {"message": "hi"}, id="object-for-string"),
        pytest.param(OBJECT_STATE, "hi", id="string-for-object"),
        pytest.param({"type": "object"}, {}, id="empty-object"),
        pytest.param({"type": "array"}, [], id="empty-array"),
    ],
)
async def test_empty_or_mistyped_llm_state_is_a_tool_error(
    jev_client: Any, state_schema: dict[str, Any], state: Any
) -> None:
    tool = create_internal_tool(_resource(state=state_schema), AsyncMock())

    message = await _tool_message(tool, _merged_call(tool, {"state": state})["args"])

    assert message.status == "error"
    jev_client.asystem_one.assert_not_awaited()


class StateInput(BaseModel):
    ticket: dict[str, Any] = {"id": "A-104", "lines": [1, "two", {"three": 3}]}
    orderId: str = "A-104"
    customerName: str = "Ana"
    message: dict[str, str] = {"from": "customer", "text": "Charged twice"}


@pytest.mark.parametrize(
    ("state_schema", "state_props", "expected"),
    [
        pytest.param(
            {"type": "object"},
            _static({"message": "Charged twice", "meta": {"retries": None}}),
            {"message": "Charged twice", "meta": {"retries": None}},
            id="static-object",
        ),
        pytest.param(
            {"type": "array"},
            _static([{"a": 1}, {"b": "x"}]),
            [{"a": 1}, {"b": "x"}],
            id="static-array-heterogeneous-items",
        ),
        pytest.param(
            {"type": "object"},
            _argument("$['ticket']"),
            {"id": "A-104", "lines": [1, "two", {"three": 3}]},
            id="argument-object",
        ),
    ],
)
async def test_whole_state_value_reaches_jev_unchanged(
    jev_client: Any,
    state_schema: dict[str, Any],
    state_props: dict[str, Any],
    expected: Any,
) -> None:
    tool = create_internal_tool(
        _resource({"$['state']": state_props, **STATIC_QUESTIONS}, state=state_schema),
        AsyncMock(),
    )

    # The LLM sees the whole value pinned as JSON text and cannot shape it.
    assert _llm_schema(_llm_tool(tool, StateInput()), "state")["enum"] == [
        json.dumps(expected)
    ]
    call = _merged_call(tool, {"state": "ignored"}, StateInput())
    await tool.ainvoke(call["args"])

    state, _ = jev_client.asystem_one.await_args.args
    assert state == expected


async def test_object_builder_state_merges_configured_and_llm_properties(
    jev_client: Any,
) -> None:
    tool = create_internal_tool(
        _resource(
            {
                "$['state']": {"variant": "objectBuilder"},
                "$['state']['order_id']": _argument("$['orderId']"),
                "$['state']['customer']": {
                    "variant": "textBuilder",
                    "tokens": [
                        {"type": "simpleText", "rawString": "Customer "},
                        {"type": "variable", "rawString": "input.customerName"},
                    ],
                    "isSensitive": False,
                },
                "$['state']['tags']": {"variant": "arrayBuilder"},
                "$['state']['tags'][0]": _static("billing"),
                **STATIC_QUESTIONS,
            },
            state=OBJECT_STATE,
        ),
        AsyncMock(),
    )
    llm_tool = _llm_tool(tool, StateInput())
    # Configured properties are pinned; the dynamic ones stay open for the LLM.
    assert _llm_schema(llm_tool, "state", "order_id")["enum"] == ["A-104"]
    assert _llm_schema(llm_tool, "state", "customer")["enum"] == ["Customer Ana"]
    assert "enum" not in _llm_schema(llm_tool, "state", "message")
    call = _merged_call(
        tool,
        {"state": {"message": "Charged twice", "amount_usd": 49}},
        StateInput(),
    )

    await tool.ainvoke(call["args"])

    state, _ = jev_client.asystem_one.await_args.args
    assert state == {
        "message": "Charged twice",
        "amount_usd": 49,
        "order_id": "A-104",
        "customer": "Customer Ana",
        "tags": ["billing"],
    }


async def test_array_builder_fills_the_state_list(jev_client: Any) -> None:
    tool = create_internal_tool(
        _resource(
            {
                "$['state']": {"variant": "arrayBuilder"},
                "$['state'][0]": _static(
                    {"from": "system", "text": "Refund policy applies"}
                ),
                "$['state'][1]": _argument("$['message']"),
                **STATIC_QUESTIONS,
            },
            state=MESSAGES_STATE,
        ),
        AsyncMock(),
    )

    await tool.ainvoke(_merged_call(tool, {"state": []}, StateInput())["args"])

    state, _ = jev_client.asystem_one.await_args.args
    assert state == [
        {"from": "system", "text": "Refund policy applies"},
        {"from": "customer", "text": "Charged twice"},
    ]
