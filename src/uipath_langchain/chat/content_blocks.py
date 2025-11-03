from typing import Literal, Optional, List, Any, Dict, Annotated, Union

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    type: Literal["text"]
    text: str = Field(alias="text")
    annotations: Optional[List[Any]] = Field(default=None, alias="annotations")
    extras: Optional[Dict[str, Any]] = Field(default=None, alias="extras")

class ReasoningContent(BaseModel):
    type: Literal["reasoning"]
    reasoning: str = Field(alias="reasoning")
    extras: Optional[Dict[str, Any]] = Field(default=None, alias="extras")

class ImageContent(BaseModel):
    type: Literal["image"]
    url: Optional[str] = Field(default=None, alias="url")
    base64: Optional[str] = Field(default=None, alias="base64")
    id: Optional[str] = Field(default=None, alias="id")
    mime_type: Optional[str] = Field(default=None, alias="mime_type")

class AudioContent(BaseModel):
    type: Literal["audio"]
    url: Optional[str] = Field(default=None, alias="url")
    base64: Optional[str] = Field(default=None, alias="base64")
    id: Optional[str] = Field(default=None, alias="id")
    mime_type: Optional[str] = Field(default=None, alias="mime_type")

class VideoContent(BaseModel):
    type: Literal["video"]
    url: Optional[str] = Field(default=None, alias="url")
    base64: Optional[str] = Field(default=None, alias="base64")
    id: Optional[str] = Field(default=None, alias="id")
    mime_type: Optional[str] = Field(default=None, alias="mime_type")

class FileContent(BaseModel):
    type: Literal["file"]
    url: Optional[str] = Field(default=None, alias="url")
    base64: Optional[str] = Field(default=None, alias="base64")
    id: Optional[str] = Field(default=None, alias="id")
    mime_type: Optional[str] = Field(default=None, alias="mime_type")

class PlainTextContent(BaseModel):
    type: Literal["text-plain"]
    text: Optional[str] = Field(default=None, alias="text-plain")
    mime_type: Optional[str] = Field(default=None, alias="mime_type")

class ToolCallContent(BaseModel):
    type: Literal["tool_call"]
    name: str = Field(alias="name")
    args: Dict[str, Any] = Field(alias="args")
    id: str = Field(alias="id")

class ToolCallChunkContent(BaseModel):
    type: Literal["tool_call_chunk"]
    name: Optional[str] = Field(default=None, alias="name")
    args: Optional[str] = Field(default=None, alias="args")
    id: Optional[str] = Field(default=None, alias="id")
    index: Optional[int | str] = Field(default=None, alias="index")

class InvalidToolCallContent(BaseModel):
    type: Literal["invalid_tool_call"]
    name: Optional[str] = Field(default=None, alias="name")
    args: Optional[Dict[str, Any]] = Field(default=None, alias="args")
    error: Optional[str] = Field(default=None, alias="error")

class ServerToolCallContent(BaseModel):
    type: Literal["server_tool_call"]
    id: str = Field(alias="id")
    name: str = Field(alias="name")
    args: Dict[str, Any] = Field(default=None, alias="args")

class ServerToolCallChunkContent(BaseModel):
    type: Literal["server_tool_call_chunk"]
    id: Optional[str] = Field(default=None, alias="id")
    name: Optional[str] = Field(default=None, alias="name")
    args: Optional[Dict[str, Any]] = Field(default=None, alias="args")
    index: Optional[int | str] = Field(default=None, alias="index")

class ServerToolResultContent(BaseModel):
    type: Literal["server_tool_result"]
    toll_call_id: str = Field(alias="toll_call_id")
    id: Optional[str] = Field(default=None, alias="id")
    status: str = Field(alias="status")
    output: Optional[Any] = Field(default=None, alias="output")

ContentBlock = Annotated[
    Union[
        TextContent,
        ReasoningContent,
        ImageContent,
        AudioContent,
        VideoContent,
        FileContent,
        PlainTextContent,
        ToolCallContent,
        ToolCallChunkContent,
        InvalidToolCallContent,
        ServerToolCallContent,
        ServerToolCallChunkContent,
        ServerToolResultContent,
        InvalidToolCallContent,
        ServerToolResultContent
    ],
    Field(discriminator="type")
]

