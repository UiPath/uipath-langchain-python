"""Memory retrieval and prompt injection for Agent Episodic Memory.

At agent execution start, queries the Memory resource for similar past episodes
using semantic search on the agent's inputs. Retrieved memories are formatted as
few-shot examples and injected into the system prompt.

Uses a LangChain BaseRetriever to integrate with the standard retrieval interface.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from uipath.platform import UiPath
from uipath.platform.memory import (
    MemoryMatch,
    MemoryMatchField,
    MemorySearchRequest,
    MemorySearchResponse,
    SearchField,
    SearchMode,
    SearchSettings,
)

logger = logging.getLogger(__name__)

# Default memory retrieval settings
DEFAULT_RESULT_COUNT = 5
DEFAULT_THRESHOLD = 0.7

MEMORY_INJECTION_TEMPLATE = """

## Relevant Examples from Past Executions

The following are similar input/output pairs from past agent executions. Use these examples to inform your response:

{examples}"""

EXAMPLE_TEMPLATE = """<example>
Inputs: {inputs}
Outputs: {outputs}
</example>"""


class UiPathMemoryRetriever(BaseRetriever):
    """LangChain retriever that queries UiPath Agent Episodic Memory.

    Wraps the UiPath MemoryService.search() to provide a standard
    LangChain retriever interface for few-shot memory recall.
    """

    memory_space_id: str = Field(description="ID of the memory space to search.")
    result_count: int = Field(
        default=DEFAULT_RESULT_COUNT, description="Max results to return."
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD, description="Min similarity score."
    )
    search_mode: SearchMode = Field(
        default=SearchMode.Semantic, description="Search mode to use."
    )
    folder_key: Optional[str] = Field(
        default=None, description="Optional folder key for the memory resource."
    )

    _sdk: UiPath = PrivateAttr()

    def __init__(self, sdk: UiPath, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sdk = sdk

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve relevant memory documents for a query string.

        Args:
            query: Serialized input arguments as a search query.
            run_manager: Callback manager (unused).

        Returns:
            List of Documents, each representing a past execution example.
        """
        request = MemorySearchRequest(
            fields=[SearchField(key_path=["input"], value=query)],
            settings=SearchSettings(
                result_count=self.result_count,
                threshold=self.threshold,
                search_mode=self.search_mode,
            ),
        )

        try:
            response: MemorySearchResponse = self._sdk.memory.search(
                memory_space_id=self.memory_space_id,
                request=request,
                folder_key=self.folder_key,
            )
            logger.info(
                f"Memory search returned {len(response.results)} results "
                f"for memory space '{self.memory_space_id}'"
            )
            return _matches_to_documents(response.results)
        except Exception:
            logger.warning(
                f"Failed to search memory space '{self.memory_space_id}'",
                exc_info=True,
            )
            return []


def build_memory_search_request(
    input_arguments: Dict[str, Any],
    result_count: int = DEFAULT_RESULT_COUNT,
    threshold: float = DEFAULT_THRESHOLD,
    search_mode: SearchMode = SearchMode.Semantic,
) -> MemorySearchRequest:
    """Build a memory search request from the agent's current inputs.

    Args:
        input_arguments: The agent's current input arguments (field name -> value).
        result_count: Maximum number of results to return.
        threshold: Minimum similarity threshold for results.
        search_mode: Search mode (Semantic or Hybrid).

    Returns:
        A MemorySearchRequest ready to send to the memory service.
    """
    search_fields = [
        SearchField(key_path=[name], value=str(value))
        for name, value in input_arguments.items()
        if value is not None and not name.startswith("uipath__")
    ]

    return MemorySearchRequest(
        fields=search_fields,
        settings=SearchSettings(
            result_count=result_count,
            threshold=threshold,
            search_mode=search_mode,
        ),
    )


def format_memory_results(matches: List[MemoryMatch]) -> str:
    """Format memory search results as few-shot examples for prompt injection.

    Args:
        matches: List of memory matches from a search response.

    Returns:
        Formatted string ready for injection into the system prompt.
        Returns empty string if no matches.
    """
    if not matches:
        return ""

    examples = []
    for match in matches:
        fields_str = _format_fields(match.fields)
        examples.append(
            EXAMPLE_TEMPLATE.format(inputs=fields_str, outputs=fields_str)
        )

    return MEMORY_INJECTION_TEMPLATE.format(examples="\n".join(examples))


def format_memory_documents(documents: List[Document]) -> str:
    """Format retriever Documents as few-shot examples for prompt injection.

    Args:
        documents: List of Documents returned by UiPathMemoryRetriever.

    Returns:
        Formatted string ready for injection into the system prompt.
        Returns empty string if no documents.
    """
    if not documents:
        return ""

    examples = []
    for doc in documents:
        examples.append(
            EXAMPLE_TEMPLATE.format(
                inputs=doc.metadata.get("inputs", ""),
                outputs=doc.page_content,
            )
        )

    return MEMORY_INJECTION_TEMPLATE.format(examples="\n".join(examples))


async def retrieve_memories(
    sdk: UiPath,
    memory_space_id: str,
    input_arguments: Dict[str, Any],
    folder_key: Optional[str] = None,
    result_count: int = DEFAULT_RESULT_COUNT,
    threshold: float = DEFAULT_THRESHOLD,
    search_mode: SearchMode = SearchMode.Semantic,
) -> MemorySearchResponse:
    """Query the memory service for relevant past episodes.

    Args:
        sdk: UiPath SDK instance.
        memory_space_id: ID of the memory space to search.
        input_arguments: The agent's current input arguments.
        folder_key: Optional folder key for the memory resource.
        result_count: Maximum number of results to return.
        threshold: Minimum similarity score threshold.
        search_mode: Search mode (Semantic or Hybrid).

    Returns:
        MemorySearchResponse containing matched memory items.
    """
    request = build_memory_search_request(
        input_arguments=input_arguments,
        result_count=result_count,
        threshold=threshold,
        search_mode=search_mode,
    )

    try:
        response = await sdk.memory.search_async(
            memory_space_id=memory_space_id,
            request=request,
            folder_key=folder_key,
        )
        logger.info(
            f"Memory search returned {len(response.results)} results "
            f"for memory space '{memory_space_id}'"
        )
        return response
    except Exception:
        logger.warning(
            f"Failed to search memory space '{memory_space_id}'",
            exc_info=True,
        )
        return MemorySearchResponse(results=[])


def _format_fields(fields: List[MemoryMatchField]) -> str:
    """Format a list of memory match fields as a comma-separated string."""
    return ", ".join(f"{'.'.join(f.key_path)}: {f.value}" for f in fields)


def _matches_to_documents(matches: List[MemoryMatch]) -> List[Document]:
    """Convert MemoryMatch results to LangChain Documents."""
    documents = []
    for match in matches:
        fields_str = _format_fields(match.fields)
        documents.append(
            Document(
                page_content=fields_str,
                metadata={
                    "inputs": fields_str,
                    "memory_item_id": match.memory_item_id,
                    "score": match.score,
                    "semantic_score": match.semantic_score,
                    "weighted_score": match.weighted_score,
                },
            )
        )
    return documents
