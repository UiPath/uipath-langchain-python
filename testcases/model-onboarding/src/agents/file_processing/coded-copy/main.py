from datetime import (
    datetime,
    timezone
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)
from pydantic import (
    BaseModel,
    Field
)
from pydantic import (
    ConfigDict
)
from typing import (
    Sequence
)
from uipath.agent.models.agent import (
    AgentInternalToolResourceConfig
)
from uipath.agent.react import (
    AGENT_SYSTEM_PROMPT_TEMPLATE
)
from uipath.platform.attachments import (
    Attachment
)
from uipath_langchain.agent.react import (
    create_agent
)
from uipath_langchain.agent.tools.internal_tools import (
    create_internal_tool
)
from uipath_langchain.chat.chat_model_factory import (
    get_chat_model
)
from utils import (
    interpolate_legacy_message
)

# Required alias for job attachment detection at runtime
__Job_attachment = Attachment


# LLM Model Configuration
llm = get_chat_model(
    model='gpt-5.4',
    temperature=0.0,
    max_tokens=128000,
    agenthub_config="agentsruntime",
)
    
# Context Grounding Tool: analyze_files
analyze_files_config = AgentInternalToolResourceConfig(
    name='Analyze Files',
    description='Analyze one or more files with an LLM to extract, synthesize, or answer queries about their content.',
    type='Internal',
    input_schema={'type': 'object', 'properties': {'attachments': {'type': 'array', 'items': {'$ref': '#/definitions/job-attachment'}, 'description': 'Array of files, documents, images, or other attachments to process'}, 'analysisTask': {'type': 'string', 'description': 'The task, question, or instruction for processing the files'}}, 'required': ['attachments', 'analysisTask'], 'definitions': {'job-attachment': {'type': 'object', 'properties': {'ID': {'type': 'string', 'description': 'Orchestrator attachment key'}, 'FullName': {'type': 'string', 'description': 'File name'}, 'MimeType': {'type': 'string', 'description': 'MIME type, e.g. "application/pdf", "image/png"'}, 'Metadata': {'type': 'object', 'description': 'Dictionary<string, string> of metadata', 'additionalProperties': {'type': 'string'}}}, 'required': ['ID'], 'x-uipath-resource-kind': 'JobAttachment'}}},
    output_schema={'type': 'object', 'properties': {'analysis': {'type': 'string', 'description': 'Analysis result of the attachments'}}, 'required': ['analysis']},
    properties={'requireConversationalConfirmation': False, 'toolType': 'analyze-attachments'},
    arguments={},
    argument_properties={}
)
analyze_files_tool = create_internal_tool(analyze_files_config, llm)


# Collect all tools
tools = []
tools.append(analyze_files_tool)


# Input/Output Models
class AgentInput(BaseModel):
    model_config = ConfigDict(extra='allow')
    prompt: str = Field(..., description="The task or question to answer about the attached file.")
    fileIn: Attachment


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra='allow')
    content: str | None = Field(None, description="The agent's analysis of the attached file.")

# Agent Messages Function
def create_messages(state: AgentInput) -> Sequence[SystemMessage | HumanMessage]:
    # Extract values safely from state
    fileIn = getattr(state, 'fileIn', '')
    prompt = getattr(state, 'prompt', '')

    # Apply system prompt template
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    system_prompt_content = """You are a file-processing assistant. You are given a single file (PDF or image) and a task. Use the Analyze Files tool to read the file's contents, then answer the task concisely based only on what the file contains. If the file cannot be read, say so plainly."""
    system_prompt_content = interpolate_legacy_message(system_prompt_content, state.model_dump())
    enhanced_system_prompt = (
        AGENT_SYSTEM_PROMPT_TEMPLATE
        .replace('{{systemPrompt}}', system_prompt_content)
        .replace('{{currentDate}}', current_date)
        .replace('{{agentName}}', 'Mr Assistant')
    )

    return [
        SystemMessage(content=enhanced_system_prompt),
        HumanMessage(content=interpolate_legacy_message("""{{prompt}}

{{fileIn}}""", state.model_dump())),
    ]

# Create agent graph
graph = create_agent(model=llm, messages=create_messages, tools=tools, input_schema=AgentInput, output_schema=AgentOutput)