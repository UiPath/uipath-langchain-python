from enum import StrEnum
from typing import Annotated, Any, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.reducers import add_job_attachments, merge_objects


class AgentTerminationSource(StrEnum):
    ESCALATION = "escalation"


class AgentTermination(BaseModel):
    """Agent Graph Termination model."""

    source: AgentTerminationSource
    title: str
    detail: str = ""


class InnerAgentGraphState(BaseModel):
    job_attachments: Annotated[dict[str, Attachment], add_job_attachments] = {}
    termination: AgentTermination | None = None


class AgentGraphState(BaseModel):
    """Agent Graph state for standard loop execution."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    inner_state: Annotated[InnerAgentGraphState, merge_objects] = Field(
        default_factory=InnerAgentGraphState
    )


class AgentGuardrailsGraphState(AgentGraphState):
    """Agent Guardrails Graph state for guardrail subgraph."""

    guardrail_validation_result: Optional[str] = None
    agent_result: Optional[dict[str, Any]] = None


class AgentGraphNode(StrEnum):
    INIT = "init"
    GUARDED_INIT = "guarded-init"
    AGENT = "agent"
    LLM = "llm"
    TOOLS = "tools"
    TERMINATE = "terminate"
    GUARDED_TERMINATE = "guarded-terminate"


class AgentGraphConfig(BaseModel):
    recursion_limit: int = Field(
        default=50, ge=1, description="Maximum recursion limit for the agent graph"
    )
    thinking_messages_limit: int = Field(
        default=0,
        ge=0,
        description="Max consecutive thinking messages before enforcing tool usage. 0 = force tools every time.",
    )
