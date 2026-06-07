"""SqlPromptContext — Pydantic model defining template variables for SQL prompts.

Each field has a sensible default so the prompt renders without any AgentContext.
ECP metadata and AgentContextResourceConfig override specific fields at runtime.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SqlPromptContext(BaseModel):
    """Variables available for system_prompt template rendering.

    Defaults ensure backward compatibility — the prompt works out of the box.
    ``build_prompt_context()`` fills overrides from ECP / AgentContext at runtime.
    """

    # ── Value resolution ──────────────────────────────────────────────
    value_resolution_strategy: str = Field(
        default=(
            "Verify string values by checking the field descriptions "
            "and any available metadata. Use exact casing and format."
        ),
        description=(
            "Instructs the LLM how to resolve WHERE-clause literal values. "
            "Overridden with ECP-aware guidance when allowed_values are present."
        ),
    )

    # ── Aggregation / grouping / filtering hints ──────────────────────
    aggregation_hints: str = Field(
        default=(
            "Choose aggregation fields based on their type — "
            "numeric fields for SUM/AVG, categorical fields for GROUP BY."
        ),
        description=(
            "Guidance on which fields to prefer for aggregation, grouping, "
            "and filtering. Overridden with ECP good_for_* flags when present."
        ),
    )

    # ── Domain guidance (customer-provided) ───────────────────────────
    domain_guidance: str = Field(
        default="",
        description=(
            "Free-text domain context from AgentContextResourceConfig.description. "
            "Injected as-is. Empty when the customer provides no description."
        ),
    )
