from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentGuardrailsGraphState(BaseModel):
    """Agent Guardrails Graph state for guardrail subgraph."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    guardrail_validation_result: Optional[str] = None
