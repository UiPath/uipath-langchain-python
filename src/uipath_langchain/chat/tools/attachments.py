"""Attachment resolution for conversational agents."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import (
    AgentInternalAnalyzeFilesToolProperties,
    AgentInternalToolResourceConfig,
)

from uipath_langchain.agent.tools.internal_tools.analyze_files_tool import (
    create_analyze_file_tool,
)

_ANALYZE_ATTACHMENTS_NAME = "analyze attachments"
_ANALYZE_ATTACHMENTS_DESCRIPTION = (
    "Read and interpret the content of file attachments provided by the user. "
    "Call this when you see a <uip:attachments> tag in a user message, passing "
    "the attachment objects from inside the tag and a query describing what you "
    "want to know about or do with the files."
)
_ANALYZE_ATTACHMENTS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysisTask": {
            "type": "string",
            "description": "What you want to know about or do with the files.",
        },
        "attachments": {
            "type": "array",
            "description": "The attachment objects from inside the <uip:attachments> tag.",
            "items": {
                "type": "object",
                "properties": {
                    "ID": {
                        "type": "string",
                        "description": "The unique identifier of the attachment.",
                    },
                    "FullName": {
                        "type": "string",
                        "description": "The full name of the attachment file.",
                    },
                    "MimeType": {
                        "type": "string",
                        "description": "The MIME type of the attachment.",
                    },
                },
                "required": ["ID", "FullName", "MimeType"],
            },
        },
    },
    "required": ["analysisTask", "attachments"],
}
_ANALYZE_ATTACHMENTS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysisResult": {
            "type": "string",
            "description": "The result of analyzing the file attachments.",
        },
    },
    "required": ["analysisResult"],
}


def AnalyzeAttachmentsTool(llm: BaseChatModel) -> StructuredTool:
    """Tool that reads and interprets file attachments using the provided LLM.

    The tool downloads each attachment, passes the file content to a non-streaming
    copy of the provided LLM for interpretation, and returns the result as text.
    This keeps multimodal content out of the agent's message state — the original
    ``<uip:attachments>`` metadata in HumanMessages is never modified.

    Example::

        from langchain_openai import ChatOpenAI
        from uipath_langchain.chat import AnalyzeAttachmentsTool

        llm = ChatOpenAI(model="gpt-4.1")
        tool = AnalyzeAttachmentsTool(llm=llm)
    """
    resource = AgentInternalToolResourceConfig(
        name=_ANALYZE_ATTACHMENTS_NAME,
        description=_ANALYZE_ATTACHMENTS_DESCRIPTION,
        input_schema=_ANALYZE_ATTACHMENTS_INPUT_SCHEMA,
        output_schema=_ANALYZE_ATTACHMENTS_OUTPUT_SCHEMA,
        properties=AgentInternalAnalyzeFilesToolProperties(),
    )
    return create_analyze_file_tool(resource, llm)
