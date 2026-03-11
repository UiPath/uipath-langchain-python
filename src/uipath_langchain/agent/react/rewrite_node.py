"""Question rewrite node for NL-to-SQL pipelines.

Conditionally rewrites ambiguous natural language questions into schema-aware,
unambiguous form using entity metadata and sample data. Skips rewriting when
the question is already clear and precise.
"""

import json
import logging
import os
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Overwrite

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """\
You are a query rewriter for a NL-to-SQL system.

CRITICAL OUTPUT RULES:
- You must output ONLY one of two things: the single word SKIP, or the rewritten question.
- Do NOT output any reasoning, analysis, explanations, or commentary.
- Do NOT use markdown, bullet points, or headers.
- The rewritten question must be 1-4 sentences max.

## Decision: SKIP vs REWRITE

Output SKIP when the question:
- Already uses specific terms that map clearly to schema field names
- Has explicit aggregation scope (e.g., "count of rows where X = Y")
- Has precise string filter values that likely match column values
- Is short, direct, and unambiguous about which columns/tables to use

Output the REWRITTEN question when:
- Vague terms don't match column names (e.g., "female" when schema has Sex with "M"/"F" values)
- Aggregation scope is ambiguous (e.g., "percentage of X" without denominator)
- Column references are ambiguous (e.g., "name" could be multiple columns)
- String matching needs sample data to resolve actual values
- Join conditions between entities are implicit

## Rewrite rules

When rewriting:
- Preserve original intent exactly
- Use exact entity and field names from schema
- Resolve string values using sample data (e.g., "female" -> "Sex = 'F'")
- Spell out aggregation scope and denominators
- Clarify join conditions for multi-entity questions
- Do NOT generate SQL
- Do NOT add information not in the original question"""


def _extract_text(content: Any) -> str:
    """Extract plain text from LLM response content (may be str or list of blocks)."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _emit_result(original: str, rewritten: str | None, skipped: bool) -> None:
    """Write rewrite decision to file for eval pipeline capture."""
    result_file = os.environ.get("REWRITE_RESULT_FILE")
    if not result_file:
        return
    try:
        data: dict[str, Any] = {"original": original, "skipped": skipped}
        if rewritten:
            data["rewritten"] = rewritten
        with open(result_file, "w") as f:
            json.dump(data, f)
    except OSError as e:
        logger.warning("Failed to write rewrite result file: %s", e)


def format_sample_data_context(
    sample_data: dict[str, list[dict[str, Any]]],
) -> str:
    """Format sample data rows into a markdown string for the rewrite prompt.

    Args:
        sample_data: Mapping of entity_name to list of row dicts (up to 5 rows each).

    Returns:
        Markdown-formatted string showing sample rows per entity.
    """
    if not sample_data:
        return ""

    lines = ["## Sample Data (5 rows per entity)", ""]
    for entity_name, rows in sample_data.items():
        if not rows:
            continue
        lines.append(f"### {entity_name}")
        columns = list(rows[0].keys())
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows[:5]:
            values = [str(row.get(c, "")) for c in columns]
            values = [v[:50] + "..." if len(v) > 50 else v for v in values]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    return "\n".join(lines)


def create_rewrite_node(
    model: BaseChatModel,
    schema_context: str,
    sample_data: dict[str, list[dict[str, Any]]],
) -> Callable[..., Any]:
    """Create a LangGraph node that conditionally rewrites the user question.

    The node first classifies whether the question needs rewriting. If the question
    is already clear and precise, it outputs SKIP and the original question is kept.

    Args:
        model: The LLM to use for classification and rewriting.
        schema_context: Formatted entity schema markdown string.
        sample_data: Pre-fetched sample rows per entity, keyed by entity name.

    Returns:
        An async callable suitable for use as a LangGraph node.
    """
    sample_data_context = format_sample_data_context(sample_data)

    async def rewrite_node(state: Any) -> dict[str, Any]:
        """Classify and conditionally rewrite the HumanMessage in state."""
        messages = state.messages

        human_msg_idx = None
        original_question = None
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                human_msg_idx = i
                original_question = msg.content
                break

        if human_msg_idx is None or not original_question:
            logger.warning("No HumanMessage found in state, skipping rewrite")
            return {}

        user_prompt_parts = [f"Original question: {original_question}"]
        if schema_context:
            user_prompt_parts.append(f"\n{schema_context}")
        if sample_data_context:
            user_prompt_parts.append(f"\n{sample_data_context}")

        rewrite_messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(user_prompt_parts)),
        ]

        try:
            response = await model.ainvoke(rewrite_messages)
            rewritten = _extract_text(response.content).strip()

            if not rewritten:
                logger.warning("Rewrite returned empty, keeping original question")
                return {}

            # Check if the LLM decided to skip rewriting
            if rewritten.upper() == "SKIP":
                print(f"[REWRITE] SKIP '{original_question[:80]}'")
                _emit_result(original_question, None, skipped=True)
                return {}

            logger.info(
                "Rewrote question: '%s' -> '%s'",
                original_question[:80],
                rewritten[:80],
            )
            print(
                f"[REWRITE] '{original_question[:80]}' -> '{rewritten[:80]}'"
            )
            _emit_result(original_question, rewritten, skipped=False)

            new_messages = list(messages)
            new_messages[human_msg_idx] = HumanMessage(
                content=rewritten,
                id=messages[human_msg_idx].id,
            )
            return {"messages": Overwrite(new_messages)}

        except Exception as e:
            logger.error("Rewrite failed, keeping original question: %s", e)
            return {}

    return rewrite_node
