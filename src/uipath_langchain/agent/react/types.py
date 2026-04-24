from enum import StrEnum
from typing import Annotated, Any, Hashable, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, model_validator
from uipath.agent.react import END_EXECUTION_TOOL, RAISE_ERROR_TOOL
from uipath.platform.attachments import Attachment

from uipath_langchain.agent.react.reducers import (
    merge_dicts,
    merge_objects,
)

FLOW_CONTROL_TOOLS = [END_EXECUTION_TOOL.name, RAISE_ERROR_TOOL.name]


class InnerAgentGraphState(BaseModel):
    job_attachments: Annotated[dict[str, Attachment], merge_dicts] = {}
    initial_message_count: int | None = None
    tools_storage: Annotated[dict[Hashable, Any], merge_dicts] = {}
    memory_injection: str = ""


class InnerAgentGuardrailsGraphState(InnerAgentGraphState):
    """Extended inner state for guardrails subgraph."""

    guardrail_validation_result: Optional[bool] = None
    guardrail_validation_details: Optional[str] = None
    agent_result: Optional[dict[str, Any]] = None
    hitl_task_info: Optional[Any] = {}


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
    MEMORY_RECALL = "memory_recall"


class MemoryConfig(BaseModel):
    """Configuration for Agent Episodic Memory.

    When passed to ``create_agent()``, a MEMORY_RECALL node is added before
    INIT that queries the memory service and stores the server-generated
    systemPromptInjection in ``inner_state.memory_injection``.
    """

    memory_space_id: str = Field(description="GUID of the memory space to query.")
    memory_space_name: str = Field(
        default="", description="Name of the memory space (for tracing)."
    )
    folder_key: str | None = Field(
        default=None, description="Folder key for the memory resource."
    )
    folder_path: str | None = Field(
        default=None,
        description="Folder path for the memory resource. Resolved to folder_key at runtime if folder_key is not set.",
    )
    # Defaults match FE episodic memory settings (agentEditor.ts:324-328)
    result_count: int = Field(default=3, ge=1, le=10)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    field_weights: dict[str, float] = Field(
        description=(
            "Per-field search weights. Keys are input field names, values are "
            "weights between 0.0 and 1.0. At least one field must be specified."
        ),
    )

    @model_validator(mode="after")
    def _validate_field_weights(self) -> "MemoryConfig":
        if not self.field_weights:
            raise ValueError("field_weights must contain at least one field")
        return self


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
    tool_choice: Literal["auto", "any"] = Field(
        default="auto",
        description="The tool choice to use for the LLM. 'auto' means the LLM will choose the tool, 'any' means the LLM will return multiple tool calls in a single response.",
    )
    parallel_tool_calls: bool = Field(
        default=True,
        description="Allow the LLM to return multiple tool calls in a single response.",
    )
    strict_mode: bool = Field(
        default=False,
        description="If set, the LLM will guarantee schema validation of the tool calls.",
    )
