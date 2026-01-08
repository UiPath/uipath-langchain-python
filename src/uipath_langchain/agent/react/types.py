from enum import StrEnum
from typing import Annotated, Any, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.reducers import (
    add_job_attachments,
    merge_objects,
    replace_once,
)


class AgentTerminationSource(StrEnum):
    ESCALATION = "escalation"


class AgentTermination(BaseModel):
    """Agent Graph Termination model."""

    source: AgentTerminationSource
    title: str
    detail: str = ""


class InnerAgentGraphState(BaseModel):
    job_attachments: Annotated[dict[str, Attachment], add_job_attachments] = {}
    termination: Annotated[AgentTermination | None, replace_once] = None


class AgentGraphState(BaseModel):
    """Agent Graph state for standard loop execution."""

    substates: Annotated[dict[str, Any], merge_objects] = {}
    messages: Annotated[list[AnyMessage], add_messages] = []
    inner_state: Annotated[InnerAgentGraphState, merge_objects] = Field(
        default_factory=InnerAgentGraphState
    )
    model_config = ConfigDict(extra="allow")


class SubgraphOutputModel(BaseModel):
    """Subgraph output model."""

    substates: dict[str, Any]


class AgentGuardrailsGraphState(AgentGraphState):
    """Agent Guardrails Graph state for guardrail subgraph."""

    guardrail_validation_result: Optional[str] = None
    agent_result: Optional[dict[str, Any]] = None
    tool_call_id: Optional[str] = None


class AgentGraphNode(StrEnum):
    INIT = "init"
    GUARDED_INIT = "guarded-init"
    AGENT = "agent"
    LLM = "llm"
    TOOLS = "tools"
    TERMINATE = "terminate"
    GUARDED_TERMINATE = "guarded-terminate"
    TOOL_CALL_STATE_HANDLER = "tool-call-state-handler"
    AGGREGATOR = "aggregator"


class AgentGraphConfig(BaseModel):
    recursion_limit: int = Field(
        default=50, ge=1, description="Maximum recursion limit for the agent graph"
    )
    thinking_messages_limit: int = Field(
        default=0,
        ge=0,
        description="Max consecutive thinking messages before enforcing tool usage. 0 = force tools every time.",
    )


class UiPathToolNodeInput(AgentGraphState):
    """Tool node input model."""

    tool_call_id: str
