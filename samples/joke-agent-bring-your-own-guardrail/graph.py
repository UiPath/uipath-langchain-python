"""Joke agent guarded by a Bring Your Own Guardrail (BYOG) configuration.

The agent generates a family-friendly joke, but every agent input and output is
validated by a *customer-managed* guardrail — your own vendor connected through
Integration Service and configured by an Org Admin under
``Admin -> AI Trust Layer -> Guardrails Configurations``. This sample was
validated against a harmful-content configuration; substitute your own.

The middleware references that configuration by validator name + connection id;
the credentials never appear in this project.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel
from uipath.core.guardrails import GuardrailScope

from uipath_langchain.chat import UiPathChat
from uipath_langchain.guardrails import (
    BlockAction,
    GuardrailExecutionStage,
    UiPathByoGuardrailMiddleware,
)

# The BYOG configuration to use. Both values come from
# Admin -> AI Trust Layer -> Guardrails Configurations, or from the CLI:
# `uip agent guardrails list`, which lists your BYOG configurations alongside
# the built-in validators.
# Replace them with your own configuration's validator name and connection id.
# The connection id is also declared as a binding in bindings.json so it can be
# rebound per environment at deploy time; locally this literal value is used.
BYOG_VALIDATOR_NAME = "my-harmful-content-guardrail"
BYOG_CONNECTION_ID = "my-byog-guardrail-connection"

# Optional: `validator_parameters` tunes the validator per run.
#
# The parameter ids, types and allowed values are defined by the guardrail
# *connector*, not by this SDK — read them from the validator's `Parameters`
# array in `uip agent guardrails list` (each entry carries its `Id`, `Type`,
# `Required`, `DefaultValue` and any `Options`/`KeySource`/`Min`/`Max`/`Step`)
# and pass the values through as-is. Omit the argument to fall back to those
# defaults.
#
# The example below is for the Azure Content Safety connector's `harmful_content`
# validator: it selects the four categories explicitly and flags a category when
# `severity >= threshold` on Azure's 0/2/4/6 scale, so a threshold of 4 lets
# low-severity (2) content through while still blocking 4 and 6. Categories that
# are not selected are ignored.
#
#     from uipath.platform.guardrails import (
#         EnumListParameterValue,
#         MapEnumParameterValue,
#     )
#     from uipath_langchain.guardrails.enums import HarmfulContentEntityType
#
#     HARMFUL_CONTENT_SEVERITY_THRESHOLD = 4
#
#     BYOG_VALIDATOR_PARAMETERS = [
#         EnumListParameterValue(
#             parameter_type="enum-list",
#             id="harmfulContentEntities",
#             value=[entity.value for entity in HarmfulContentEntityType],
#         ),
#         MapEnumParameterValue(
#             parameter_type="map-enum",
#             id="harmfulContentEntityThresholds",
#             value={
#                 entity.value: HARMFUL_CONTENT_SEVERITY_THRESHOLD
#                 for entity in HarmfulContentEntityType
#             },
#         ),
#     ]
#
# ...then pass `validator_parameters=BYOG_VALIDATOR_PARAMETERS` to the middleware.


class Input(BaseModel):
    """Input schema for the joke agent."""

    topic: str


class Output(BaseModel):
    """Output schema for the joke agent."""

    joke: str


llm = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)


@tool
def analyze_joke_syntax(joke: str) -> str:
    """Analyze the syntax of a joke by counting words and letters.

    Args:
        joke: The joke text to analyze

    Returns:
        A string with the analysis results showing word count and letter count
    """
    words = joke.split()
    letter_count = sum(1 for char in joke if char.isalpha())
    return f"Words number: {len(words)}\nLetters: {letter_count}"


SYSTEM_PROMPT = """You are an AI assistant designed to generate family-friendly jokes.

1. Generate a family-friendly joke based on the given topic.
2. Use the analyze_joke_syntax tool to analyze the joke's syntax.
3. Ensure your output includes the joke.

Remember to always include the 'joke' property in your output."""

agent = create_agent(
    model=llm,
    tools=[analyze_joke_syntax],
    system_prompt=SYSTEM_PROMPT,
    middleware=[
        # Customer-managed harmful-content guardrail (BYOG). PRE_AND_POST on
        # AGENT scope: the requested topic is validated before the LLM runs and
        # the produced joke is validated before it is returned. On a violation
        # BlockAction aborts the run with the vendor's verdict details.
        *UiPathByoGuardrailMiddleware(
            validator_name=BYOG_VALIDATOR_NAME,
            scopes=[GuardrailScope.AGENT],
            action=BlockAction(),
            connection_id=BYOG_CONNECTION_ID,
            # Optionally add validator_parameters=... — see the example above.
            stage=GuardrailExecutionStage.PRE_AND_POST,
            name="BYOG Harmful Content",
        ),
        # The same configuration can also guard individual tools:
        *UiPathByoGuardrailMiddleware(
            validator_name=BYOG_VALIDATOR_NAME,
            scopes=[GuardrailScope.TOOL],
            action=BlockAction(),
            connection_id=BYOG_CONNECTION_ID,
            tools=[analyze_joke_syntax],
            stage=GuardrailExecutionStage.PRE,
            name="BYOG Tool Harmful Content",
        ),
    ],
)


async def joke_node(state: Input) -> Output:
    """Convert topic to messages, call agent, and extract joke."""
    messages = [
        HumanMessage(
            content=f"Generate a family-friendly joke based on the topic: {state.topic}"
        )
    ]
    result = await agent.ainvoke({"messages": messages})
    joke = result["messages"][-1].content
    return Output(joke=joke)


builder = StateGraph(Input, input_schema=Input, output_schema=Output)
builder.add_node("joke", joke_node)
builder.add_edge(START, "joke")
builder.add_edge("joke", END)

graph = builder.compile()
