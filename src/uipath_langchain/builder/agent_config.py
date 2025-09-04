from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AgentMessageRole(str, Enum):
    """Enum for message roles"""

    SYSTEM = "System"
    USER = "User"


class AgentMessage(BaseModel):
    """Message model for agent conversations"""

    role: AgentMessageRole
    content: str

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentSettings(BaseModel):
    """Settings for agent configuration"""

    engine: str = Field(..., description="Engine type, e.g., 'basic-v1'")
    model: str = Field(..., description="LLM model identifier")
    max_tokens: int = Field(
        ..., alias="maxTokens", description="Maximum number of tokens"
    )
    temperature: float = Field(..., description="Temperature for response generation")

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentResourceType(str, Enum):
    """Enum for resource types"""

    TOOL = "tool"
    CONTEXT = "context"
    ESCALATION = "escalation"


class AgentBaseResourceConfig(BaseModel):
    """Base resource model with common properties"""

    name: str
    description: str

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentUnknownResourceConfig(AgentBaseResourceConfig):
    """Fallback for unknown or future resource types"""

    resource_type: str = Field(alias="$resourceType")

    model_config = ConfigDict(extra="allow")


class AgentToolSettings(BaseModel):
    """Settings for tool configuration"""

    max_attempts: int = Field(0, alias="maxAttempts")
    retry_delay: int = Field(0, alias="retryDelay")
    timeout: int = Field(0)

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentToolProperties(BaseModel):
    """Properties specific to tool configuration"""

    folder_path: Optional[str] = Field(None, alias="folderPath")
    process_name: Optional[str] = Field(None, alias="processName")

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentToolResourceConfig(AgentBaseResourceConfig):
    """Tool resource with tool-specific properties"""

    resource_type: Literal[AgentResourceType.TOOL] = Field(alias="$resourceType")
    type: str = Field(..., description="Tool type")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    input_schema: Dict[str, Any] = Field(
        ..., alias="inputSchema", description="Input schema for the tool"
    )
    output_schema: Dict[str, Any] = Field(
        ..., alias="outputSchema", description="Output schema for the tool"
    )
    properties: AgentToolProperties = Field(..., description="Tool-specific properties")
    settings: AgentToolSettings = Field(
        default_factory=AgentToolSettings, description="Tool settings"
    )

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentContextSettings(BaseModel):
    """Settings for context configuration"""

    result_count: int = Field(alias="resultCount")
    retrieval_mode: Literal["Semantic", "Structured"] = Field(alias="retrievalMode")
    threshold: float = Field(default=0)

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentContextResourceConfig(AgentBaseResourceConfig):
    """Context resource with context-specific properties"""

    resource_type: Literal[AgentResourceType.CONTEXT] = Field(alias="$resourceType")
    folder_path: str = Field(alias="folderPath")
    index_name: str = Field(alias="indexName")
    settings: AgentContextSettings = Field(..., description="Context settings")

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class AgentEscalationResourceConfig(AgentBaseResourceConfig):
    """Escalation resource with escalation-specific properties"""

    resource_type: Literal[AgentResourceType.ESCALATION] = Field(alias="$resourceType")

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


# Discriminated union for known types
KnownAgentResourceConfig = Annotated[
    Union[
        AgentToolResourceConfig,
        AgentContextResourceConfig,
        AgentEscalationResourceConfig,
    ],
    Field(discriminator="resource_type"),
]

# Final union includes unknowns as a catch-all
AgentResourceConfig = Union[
    KnownAgentResourceConfig,
    AgentUnknownResourceConfig,
]


class AgentConfig(BaseModel):
    """Main agent model"""

    id: str = Field(..., description="Agent id or project name")
    name: str = Field(..., description="Agent name or project name")
    input_schema: Dict[str, Any] = Field(
        ..., alias="inputSchema", description="JSON schema for input arguments"
    )
    output_schema: Dict[str, Any] = Field(
        ..., alias="outputSchema", description="JSON schema for output arguments"
    )
    messages: List[AgentMessage] = Field(
        ..., description="List of system and user messages"
    )
    features: List[Any] = Field(
        default_factory=list, description="Currently empty feature list"
    )
    version: str = Field("1.0.0", description="Agent version")
    settings: AgentSettings = Field(..., description="Agent settings configuration")
    resources: List[AgentResourceConfig] = Field(
        ..., description="List of tools, context, and escalation resources"
    )

    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )
