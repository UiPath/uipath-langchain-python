"""Agent graph construction - wrapper delegating to uipath_langchain.agent.graph."""

from typing import Any, Callable, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from uipath.agent.models.agent import (
    LowCodeAgentDefinition,
)
from uipath_langchain.agent.guardrails import build_guardrails_with_actions
from uipath_langchain.agent.react import (
    AgentGraphConfig,
    create_agent,
    resolve_input_model,
    resolve_output_model,
)
from uipath_langchain.agent.tools import create_tools_from_resources

from .llm_utils import create_llm
from .message_utils import build_agent_messages

# Maximum number of agent loop iterations before termination
# Set to 50 to prevent infinite loops while allowing complex reasoning chains
AGENT_LOOP_RECURSION_LIMIT = 50


def create_message_factory(
    agent_definition: LowCodeAgentDefinition,
) -> Callable[[Any], Sequence[SystemMessage | HumanMessage]]:
    """Create a callable that builds messages with deferred interpolation.

    This allows template variables like {{a}}, {{b}} to be interpolated
    at runtime when the actual input values are available, rather than
    at graph build time.

    Args:
        agent_definition: Agent definition containing messages with templates

    Returns:
        A callable that takes state (with input fields) and returns interpolated messages
    """

    def message_factory(state: Any) -> List[SystemMessage | HumanMessage]:
        # Extract input data from state (which is a Pydantic model or dict)
        if isinstance(state, BaseModel):
            input_data = state.model_dump()
        elif isinstance(state, dict):
            input_data = state
        else:
            input_data = {}

        return build_agent_messages(
            agent_definition.messages, input_data, agent_definition.name or ""
        )

    return message_factory


async def build_agent_graph(
    agent_definition: LowCodeAgentDefinition,
    input_data: dict[str, Any],
):
    """Build LangGraph agent from agent.json configuration and optional input data.

    Args:
        agent_definition: Agent definition model
        input_data: Optional input data for the agent (used for CLI runs, ignored for evals)

    Returns:
        StateGraph configured with the agent definition and feature flags.
    """

    tools = await create_tools_from_resources(agent_definition)
    llm = create_llm(
        model=agent_definition.settings.model,
        temperature=agent_definition.settings.temperature,
        max_tokens=agent_definition.settings.max_tokens,
    )

    input_model = resolve_input_model(agent_definition.input_schema)
    output_model = resolve_output_model(agent_definition.output_schema)

    # Use deferred message interpolation via a callable
    # This allows template variables to be resolved at runtime
    # when actual inputs are available (important for evals)
    message_factory = create_message_factory(agent_definition)

    guardrails = build_guardrails_with_actions(agent_definition.guardrails)

    # Create agent config with feature flags
    agent_config = AgentGraphConfig(
        recursion_limit=AGENT_LOOP_RECURSION_LIMIT,
    )

    return create_agent(
        model=llm,
        tools=tools,
        messages=message_factory,
        input_schema=input_model,
        output_schema=output_model,
        config=agent_config,
        guardrails=guardrails,
    )
