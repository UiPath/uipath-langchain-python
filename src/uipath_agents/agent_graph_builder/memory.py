"""Memory retrieval and prompt injection for Agent Episodic Memory.

At agent execution start, queries the Memory resource for similar past episodes
using semantic search on the agent's inputs. Retrieved memories are formatted as
few-shot examples and injected into the system prompt.
"""

import logging
from typing import Any, Dict, List, Optional

from uipath.platform import UiPath
from uipath.platform.memory import (
    MemoryField,
    MemoryQueryRequest,
    MemoryQueryResponse,
    MemoryQueryResult,
)

logger = logging.getLogger(__name__)

# Default memory retrieval settings
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.7

MEMORY_INJECTION_TEMPLATE = """

## Relevant Examples from Past Executions

The following are similar input/output pairs from past agent executions. Use these examples to inform your response:

{examples}"""

EXAMPLE_TEMPLATE = """<example>
Inputs: {inputs}
Outputs: {outputs}
</example>"""


def build_memory_query(
    input_arguments: Dict[str, Any],
    input_weights: Optional[Dict[str, float]] = None,
    partition_key: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    threshold: Optional[float] = DEFAULT_THRESHOLD,
) -> MemoryQueryRequest:
    """Build a memory query request from the agent's current inputs.

    Args:
        input_arguments: The agent's current input arguments (field name -> value).
        input_weights: Optional mapping of field names to relevance weights.
            Fields not in this mapping are included with default weight.
        partition_key: Optional scoping key to restrict memory recall.
        top_k: Maximum number of results to return.
        threshold: Minimum similarity threshold for results.

    Returns:
        A MemoryQueryRequest ready to send to the ECS memory service.
    """
    inputs = [
        MemoryField(fieldName=name, fieldValue=str(value))
        for name, value in input_arguments.items()
        if value is not None and not name.startswith("uipath__")
    ]

    return MemoryQueryRequest(
        inputs=inputs,
        topK=top_k,
        threshold=threshold,
        partitionKey=partition_key,
    )


def format_memory_results(results: List[MemoryQueryResult]) -> str:
    """Format memory query results as few-shot examples for prompt injection.

    Args:
        results: List of memory query results with scores.

    Returns:
        Formatted string ready for injection into the system prompt.
        Returns empty string if no results.
    """
    if not results:
        return ""

    examples = []
    for result in results:
        item = result.memory_item
        inputs_str = ", ".join(
            f"{f.field_name}: {f.field_value}" for f in item.inputs
        )
        outputs_str = ", ".join(
            f"{f.field_name}: {f.field_value}" for f in item.outputs
        )
        examples.append(
            EXAMPLE_TEMPLATE.format(inputs=inputs_str, outputs=outputs_str)
        )

    return MEMORY_INJECTION_TEMPLATE.format(examples="\n".join(examples))


async def retrieve_memories(
    sdk: UiPath,
    memory_resource_name: str,
    input_arguments: Dict[str, Any],
    folder_key: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    threshold: Optional[float] = DEFAULT_THRESHOLD,
    partition_key: Optional[str] = None,
) -> MemoryQueryResponse:
    """Query the memory service for relevant past episodes.

    Args:
        sdk: UiPath SDK instance.
        memory_resource_name: Name of the memory resource to query.
        input_arguments: The agent's current input arguments.
        folder_key: Optional folder key for the memory resource.
        top_k: Maximum number of results to return.
        threshold: Minimum similarity score threshold.
        partition_key: Optional scoping key for partitioned memory.

    Returns:
        MemoryQueryResponse containing matched memory items.
    """
    query = build_memory_query(
        input_arguments=input_arguments,
        partition_key=partition_key,
        top_k=top_k,
        threshold=threshold,
    )

    try:
        response = sdk.memory.query(
            name=memory_resource_name,
            request=query,
            folder_key=folder_key,
        )
        logger.info(
            f"Memory query returned {len(response.results)} results "
            f"for resource '{memory_resource_name}'"
        )
        return response
    except Exception:
        logger.warning(
            f"Failed to query memory resource '{memory_resource_name}'",
            exc_info=True,
        )
        return MemoryQueryResponse(results=[])
