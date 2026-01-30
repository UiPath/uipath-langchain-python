"""Context grounding span attribute classes."""

from typing import Any, Optional

from pydantic import ConfigDict, Field

from .base import BaseSpanAttributes
from .types import SpanType


class ContextGroundingToolSpanAttributes(BaseSpanAttributes):
    """Attributes for context grounding tool spans."""

    model_config = ConfigDict(populate_by_name=True)

    retrieval_mode: str = Field(..., alias="retrieval_mode")
    query: str = Field(..., alias="query")
    threshold: Optional[float] = Field(None, alias="threshold")
    number_of_results: Optional[int] = Field(None, alias="number_of_results")
    filter: Optional[str] = Field(None, alias="filter")
    folder_path_prefix: Optional[str] = Field(None, alias="folder_path_prefix")
    file_extension: Optional[str] = Field(None, alias="file_extension")
    is_system_index: Optional[bool] = Field(None, alias="system_index")
    results: Optional[Any] = Field(None, alias="results")
    output_columns: Optional[Any] = Field(None, alias="output_columns")
    web_search_grounding: Optional[bool] = Field(None, alias="web_search_grounding")
    citation_mode: Optional[str] = Field(None, alias="citation_mode")
    index_id: Optional[str] = Field(None, alias="index_id")

    @property
    def type(self) -> str:
        return SpanType.CONTEXT_GROUNDING_TOOL
