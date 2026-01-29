from enum import StrEnum
from typing import Annotated, Any, Hashable, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.reducers import (
    merge_dicts,
    merge_objects,
)
from uipath_langchain.chat.types import APIFlavor, LLMProvider

FLOW_CONTROL_TOOLS = [END_EXECUTION_TOOL.name, RAISE_ERROR_TOOL.name]


class AgentSettings(BaseModel):
    """Agent settings extracted from the LLM model."""

    llm_provider: LLMProvider
    api_flavor: APIFlavor


class InnerAgentGraphState(BaseModel):
    job_attachments: Annotated[dict[str, Attachment], merge_dicts] = {}
    agent_settings: AgentSettings | None = None
    tools_storage: Annotated[dict[Hashable, Any], merge_dicts] = {}


class InnerAgentGuardrailsGraphState(InnerAgentGraphState):
    """Extended inner state for guardrails subgraph."""

    guardrail_validation_result: Optional[bool] = None
    guardrail_validation_details: Optional[str] = None
    agent_result: Optional[dict[str, Any]] = None


class AgentGraphState(BaseModel):
    """Agent Graph state for standard loop execution."""

    messages: Annotated[list[AnyMessage], add_messages] = []
    inner_state: Annotated[InnerAgentGraphState, merge_objects] = Field(
        default_factory=InnerAgentGraphState
    )


class AgentGuardrailsGraphState(AgentGraphState):
    """Agent Guardrails Graph state for guardrail subgraph."""

    inner_state: Annotated[InnerAgentGuardrailsGraphState, merge_objects] = Field(
        default_factory=InnerAgentGuardrailsGraphState
    )


class AgentGraphNode(StrEnum):
    INIT = "init"
    GUARDED_INIT = "guarded-init"
    AGENT = "agent"
    LLM = "llm"
    TOOLS = "tools"
    TERMINATE = "terminate"
    GUARDED_TERMINATE = "guarded-terminate"


class AgentGraphConfig(BaseModel):
    llm_messages_limit: int = Field(
        default=25,
        ge=1,
        description="Maximum number of LLM calls allowed per agent execution",
    )
    thinking_messages_limit: int = Field(
        default=0,
        ge=0,
        description="Max consecutive thinking messages before enforcing tool calling. 0 = force tool calling every time.",
    )
    is_conversational: bool = Field(
        default=False, description="If set, creates a graph for conversational agents"
    )
