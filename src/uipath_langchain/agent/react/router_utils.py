"""Routing functions for conditional edges in the agent graph."""


from langchain_core.messages import AIMessage, AnyMessage

from ..exceptions import AgentNodeRoutingException


def validate_last_message_is_AI(messages: list[AnyMessage]) -> AIMessage:
    """Validate and return last message from state.

    Raises:
        AgentNodeRoutingException: If messages are empty or last message is not AIMessage
    """
    if not messages:
        raise AgentNodeRoutingException(
            "No messages in state - cannot route after agent"
        )

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        raise AgentNodeRoutingException(
            f"Last message is not AIMessage (type: {type(last_message).__name__}) - cannot route after agent"
        )

    return last_message
