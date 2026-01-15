"""Message creation and template interpolation utilities."""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Pattern, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentEscalationResourceConfig,
    AgentMessage,
    AgentMessageRole,
    BaseAgentResourceConfig,
    BaseAgentToolResourceConfig,
    LowCodeAgentDefinition,
)
from uipath.agent.react import (
    AGENT_SYSTEM_PROMPT_TEMPLATE,
    PromptUserSettings,
    get_chat_system_prompt,
)
from uipath.agent.utils.text_tokens import (
    build_string_from_tokens,
    safe_get_nested,
    serialize_argument,
)

logger = logging.getLogger(__name__)

# Whitelist pattern: allow alphanumeric, dots, underscores, and array notation []*
SAFE_ARGUMENT_PATTERN: Pattern[str] = re.compile(r"^[a-zA-Z0-9_.\[\]*]+$")


def apply_system_prompt_template(system_prompt_content: str, agent_name: str) -> str:
    """Apply AGENT_SYSTEM_PROMPT_TEMPLATE with systemPrompt, currentDate, and agentName."""
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return (
        AGENT_SYSTEM_PROMPT_TEMPLATE.replace("{{systemPrompt}}", system_prompt_content)
        .replace("{{currentDate}}", current_date)
        .replace("{{agentName}}", agent_name)
    )


def build_agent_messages(
    agent_messages: List[AgentMessage],
    input_arguments: dict[str, Any],
    agent_name: str,
    tool_names: list[str] | None = None,
    escalation_names: list[str] | None = None,
    context_names: list[str] | None = None,
) -> list[SystemMessage | HumanMessage]:
    """Convert agent messages to LangChain messages with variable interpolation.

    System messages are wrapped in AGENT_SYSTEM_PROMPT_TEMPLATE with currentDate and agentName.
    Expects exactly one system message and one user message.

    Raises:
        ValueError: If system or user message is missing
    """
    system_message = next((msg for msg in agent_messages if msg.role == "system"), None)
    if system_message is None:
        raise ValueError("Agent configuration must contain exactly one system message")

    user_message = next((msg for msg in agent_messages if msg.role == "user"), None)
    if user_message is None:
        raise ValueError("Agent configuration must contain exactly one user message")

    system_prompt_content = _build_message_content(
        system_message, input_arguments, tool_names, escalation_names, context_names
    )
    system_prompt = apply_system_prompt_template(system_prompt_content, agent_name)
    user_prompt = _build_message_content(
        user_message, input_arguments, tool_names, escalation_names, context_names
    )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def extract_input_data_from_state(
    state: BaseModel | dict[str, Any],
    input_model: type[BaseModel],
) -> dict[str, Any]:
    """Extract only input schema fields from graph state, filtering out internal fields.

    This prevents internal LangGraph state fields (messages, termination, agent_outcome, etc.)
    from leaking into template interpolation.

    Args:
        state: The combined agent graph state (InnerAgentGraphState = AgentGraphState + input_schema).
               At runtime, this is a dynamically created class that inherits from both.
        input_model: The input schema model defining allowed fields

    Returns:
        Dictionary containing only graph input arguments defined in the agent's input_schema
    """
    if isinstance(state, BaseModel):
        graph_state = state.model_dump()
    else:
        graph_state = state
    return input_model.model_validate(graph_state, from_attributes=True).model_dump()


def create_message_factory(
    agent_definition: LowCodeAgentDefinition,
    input_model: type[BaseModel],
) -> Callable[[BaseModel | dict[str, Any]], Sequence[SystemMessage | HumanMessage]]:
    """Create a callable that builds messages at runtime using interpolation.

    Args:
        agent_definition: Agent definition containing messages with templates
        input_model: The input schema model to extract field names from

    Returns:
        A callable that takes the agent's graph state and returns interpolated messages.
        InnerAgentGraphState is created dynamically as: type("InnerAgentGraphState", (AgentGraphState, input_schema), {})
    """

    def message_factory(
        state: BaseModel | dict[str, Any],
    ) -> List[SystemMessage | HumanMessage]:
        input_arguments = extract_input_data_from_state(state, input_model)
        return _create_messages_from_definition(agent_definition, input_arguments)

    def conversational_message_factory(
        state: BaseModel | dict[str, Any],
    ) -> List[SystemMessage | HumanMessage]:
        input_arguments = extract_input_data_from_state(state, input_model)
        return build_conversational_agent_messages(agent_definition, input_arguments)

    return (
        conversational_message_factory
        if agent_definition.is_conversational
        else message_factory
    )


def _create_messages_from_definition(
    agent_definition: LowCodeAgentDefinition, input_arguments: dict[str, Any]
) -> list[SystemMessage | HumanMessage]:
    def extract_resource_names(
        resources: Sequence[BaseAgentResourceConfig],
        resource_type: type[BaseAgentResourceConfig],
    ) -> list[str]:
        """Collect the names of resources of the given type."""
        return [r.name for r in resources if isinstance(r, resource_type)]

    """Create messages from agent definition and input data."""
    tool_names = extract_resource_names(
        agent_definition.resources, BaseAgentToolResourceConfig
    )
    escalation_names = extract_resource_names(
        agent_definition.resources, AgentEscalationResourceConfig
    )
    context_names = extract_resource_names(
        agent_definition.resources, AgentContextResourceConfig
    )

    return build_agent_messages(
        agent_definition.messages,
        input_arguments,
        agent_definition.name or "",
        tool_names=tool_names,
        escalation_names=escalation_names,
        context_names=context_names,
    )


def _build_message_content(
    message: AgentMessage,
    input_arguments: dict[str, Any],
    tool_names: list[str] | None = None,
    escalation_names: list[str] | None = None,
    context_names: list[str] | None = None,
) -> str:
    """Build a prompt from an AgentMessage, handling both legacy content and new content_tokens."""
    if message.content_tokens is None:
        # Support both {{input.x}} (Studio Web) and {{x}} (direct) formats
        wrapped_input = {"input": input_arguments, **input_arguments}
        return interpolate_legacy_message(message.content or "", wrapped_input)

    return build_string_from_tokens(
        message.content_tokens,
        input_arguments,
        tool_names,
        escalation_names,
        context_names,
    )


def interpolate_legacy_message(content: str, input_values: dict[str, Any]) -> str:
    """Replace {{variable}} placeholders with values. Supports nested paths (e.g., {{user.email}}).

    Uses whitelist pattern to prevent injection attacks.
    """
    argument_placeholder_pattern = r"\{\{([^}]+)\}\}"
    matches = re.findall(argument_placeholder_pattern, content)

    interpolated = content
    for field_path in matches:
        field_path = field_path.strip()

        # Validate field path to prevent injection
        if not SAFE_ARGUMENT_PATTERN.match(field_path):
            logger.warning(
                f"Skipping unsafe placeholder '{{{{{field_path}}}}}' - "
                f"only alphanumeric, dots, and underscores are allowed"
            )
            continue

        value = safe_get_nested(input_values, field_path)

        if value is not None:
            placeholder = f"{{{{{field_path}}}}}"
            interpolated = interpolated.replace(placeholder, serialize_argument(value))

    return interpolated


def build_conversational_agent_messages(
    agent_definition: LowCodeAgentDefinition, input_arguments: dict[str, Any]
) -> list[SystemMessage | HumanMessage]:
    # See note in runtime/factory.py about the use of input arguments for user settings.
    user_settings = extract_user_settings(input_arguments)
    if user_settings is None:
        logger.warning(
            "user_settings property not provided in input - user context will not be included in system prompt"
        )

    system_message: AgentMessage | None = next(
        (
            msg
            for msg in agent_definition.messages
            if msg.role == AgentMessageRole.SYSTEM
        ),
        None,
    )
    if system_message is None:
        raise ValueError(
            "Conversational agent configuration must contain exactly one system message"
        )

    system_prompt = get_chat_system_prompt(
        agent_name=agent_definition.name,
        model=agent_definition.settings.model,
        system_message=system_message.content,
        user_settings=user_settings,
    )

    # Note that we ignore any user messages in the agent definition's messages array. Currently agent builder is leaving
    # a vestigial "What is the current date?" user message in the agent definition. For conversational agents the input
    # message comes from the input arguments and are added to the initial state by UiPathLangGraphRuntime. The init node
    # takes care of ensuring the system message is first, followed by the initial user message.

    return [SystemMessage(content=system_prompt)]


def extract_user_settings(
    input_data: Optional[dict[str, Any]],
) -> Optional[PromptUserSettings]:
    """Extract user settings from input data.

    Args:
        input_data: Input data dict that may contain userSettings
        is_resume: Whether this is a resume operation (suppresses warning)

    Returns:
        PromptUserSettings if found in input, None otherwise
    """
    if not input_data or not isinstance(input_data, dict):
        return None

    user_settings_data = input_data.get("userSettings")
    if not user_settings_data or not isinstance(user_settings_data, dict):
        return None

    return PromptUserSettings(
        name=user_settings_data.get("name"),
        email=user_settings_data.get("email"),
        role=user_settings_data.get("role"),
        department=user_settings_data.get("department"),
        company=user_settings_data.get("company"),
        country=user_settings_data.get("country"),
        timezone=user_settings_data.get("timezone"),
    )
