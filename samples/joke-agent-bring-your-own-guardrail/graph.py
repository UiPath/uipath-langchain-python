"""Joke agent guarded by a Bring Your Own Guardrail (BYOG) configuration.

One agent, one BYOG configuration, **both guardrail flavors**:

- **Middleware** (``UiPathByoGuardrailMiddleware``) guards the AGENT scope:
  the requested topic and the produced joke are validated and violations are
  logged (``LogAction``) -- the run continues.
- **Decorator** (``@guardrail`` + ``ByoValidator``) guards the LLM scope:
  every prompt is validated before it reaches the model and a violation
  blocks the run (``BlockAction``).

The validator is *customer-managed* — your own vendor connected through
Integration Service and configured by an Org Admin under
``Admin -> AI Trust Layer -> Guardrails Configurations``. Both flavors
reference that configuration purely by validator name + connection id; the
credentials never appear in this project. This sample was validated against a
harmful-content configuration; substitute your own.
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
    ByoValidator,
    GuardrailExecutionStage,
    LogAction,
    UiPathByoGuardrailMiddleware,
    guardrail,
)

# The BYOG configuration both flavors use. Both values come from
# Admin -> AI Trust Layer -> Guardrails Configurations, or from the CLI:
# `uip agent guardrails list --byo` (fields `ByoValidatorName` and
# `ByoConnectionId`).
# Replace them with your own configuration's values.
BYOG_VALIDATOR_NAME = "my-harmful-content-guardrail"
BYOG_CONNECTION_ID = "my-byog-guardrail-connection"

# Optional: `validator_parameters` (middleware) / `parameters` (ByoValidator)
# tune the validator per run.
#
# The parameter ids, types and allowed values are defined by the guardrail
# *connector*, not by this SDK — read them from the validator's `Parameters`
# array in `uip agent guardrails list --byo` (each entry carries its `Id`,
# `Type`, `Required`, `DefaultValue` and any `Options`/`KeySource`/`Min`/`Max`/
# `Step`) and pass the values through as-is. Omit the argument to fall back to
# those defaults.
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
# ...then pass `validator_parameters=BYOG_VALIDATOR_PARAMETERS` to the
# middleware, or `parameters=BYOG_VALIDATOR_PARAMETERS` to `ByoValidator`.

# Decorator flavor: a reusable validator object referencing the same BYOG
# configuration — declare once, stack on any number of targets.
byog_harmful_content = ByoValidator(
    BYOG_VALIDATOR_NAME,
    connection_id=BYOG_CONNECTION_ID,
)


class Input(BaseModel):
    """Input schema for the joke agent."""

    topic: str


class Output(BaseModel):
    """Output schema for the joke agent."""

    joke: str


# Decorator flavor guards the LLM. LLM scope is inferred from the factory's
# BaseChatModel return value; PRE only: every prompt is validated by the
# customer's vendor before it reaches the model, and a violation aborts the
# run (BlockAction) with the vendor's verdict details.
@guardrail(
    validator=byog_harmful_content,
    action=BlockAction(),
    stage=GuardrailExecutionStage.PRE,
    name="BYOG LLM Harmful Content",
)
def create_llm():
    """Create the LLM guarded by the BYOG configuration."""
    return UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)


llm = create_llm()


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
        # Middleware flavor guards the agent. PRE_AND_POST on AGENT scope: the
        # requested topic and the produced joke are validated by the customer's
        # vendor, and a violation is logged (LogAction) without stopping the
        # run -- the LLM-scope BlockAction above is the enforcing guardrail.
        *UiPathByoGuardrailMiddleware(
            validator_name=BYOG_VALIDATOR_NAME,
            scopes=[GuardrailScope.AGENT],
            action=LogAction(),
            connection_id=BYOG_CONNECTION_ID,
            # Optionally add validator_parameters=... — see the example above.
            stage=GuardrailExecutionStage.PRE_AND_POST,
            name="BYOG Harmful Content",
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
