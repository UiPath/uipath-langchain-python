"""Prompt version registry and context builder.

A ``PromptVersion`` pairs a name with a Python format-string template. The
template is rendered against the values of a :class:`SqlPromptContext` via
``template.format_map(ctx.model_dump())`` — fields not referenced by a given
template are simply ignored, which lets older versions (e.g. ``v0``) coexist
with newer ones that take more variables.

``build_prompt_context`` resolves overrides for ``SqlPromptContext`` from the
runtime entity list and the resource description, falling back to the
Pydantic defaults declared on the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .context import SqlPromptContext
from .v0 import TEMPLATE as V0_TEMPLATE
from .v1 import TEMPLATE as V1_TEMPLATE

if TYPE_CHECKING:
    from uipath.platform.entities import Entity


@dataclass(frozen=True)
class PromptVersion:
    """A named SQL prompt template rendered against a :class:`SqlPromptContext`."""

    name: str
    template: str

    def render(self, ctx: SqlPromptContext) -> str:
        """Render the template using the values from ``ctx``."""
        return self.template.format_map(ctx.model_dump())


_REGISTRY: dict[str, PromptVersion] = {
    "v0": PromptVersion(name="v0", template=V0_TEMPLATE),
    "v1": PromptVersion(name="v1", template=V1_TEMPLATE),
}

DEFAULT_PROMPT_VERSION = "v1"


def get_prompt_version(name: str | None = None) -> PromptVersion:
    """Return the ``PromptVersion`` for ``name`` (or the default)."""
    key = name or DEFAULT_PROMPT_VERSION
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown prompt version {key!r}. Available versions: {valid}."
        ) from exc


def list_prompt_versions() -> list[str]:
    """Return the registered prompt version names."""
    return sorted(_REGISTRY)


def _entities_have_ecp_metadata(entities: Iterable[Entity]) -> bool:
    """Detect whether any entity field carries ECP-style metadata.

    Returns True if at least one field declares ``allowed_values``,
    ``examples``, or any of the ``good_for_*`` flags. Treated as a soft
    signal — when False, generic guidance is used.
    """
    ecp_attrs = (
        "allowed_values",
        "examples",
        "good_for_aggregation",
        "good_for_grouping",
        "good_for_filtering",
    )
    for entity in entities:
        for field in entity.fields or []:
            for attr in ecp_attrs:
                value = getattr(field, attr, None)
                if value:
                    return True
    return False


def build_prompt_context(
    entities: Iterable[Entity] | None = None,
    resource_description: str | None = None,
) -> SqlPromptContext:
    """Build a :class:`SqlPromptContext` from runtime inputs.

    Args:
        entities: Resolved Data Fabric entities. When any field carries ECP
            metadata, value-resolution and aggregation hints are upgraded
            from the generic defaults to ECP-aware guidance.
        resource_description: Free-text description from
            ``AgentContextResourceConfig.description``. Wrapped under a
            ``## Domain Guidance`` section when non-empty.

    Returns:
        A :class:`SqlPromptContext` with overrides applied. Fields not
        overridden fall back to the Pydantic defaults declared on the model.
    """
    overrides: dict[str, str] = {}

    entity_list = list(entities) if entities is not None else []
    if entity_list and _entities_have_ecp_metadata(entity_list):
        overrides["value_resolution_strategy"] = (
            "Match WHERE values against the **allowed_values** and "
            "**examples** listed for each field in the entity metadata "
            "above. Use the exact stored form (casing, punctuation, "
            "abbreviations). Do NOT issue a runtime probe — trust the "
            "metadata. If a value is not in allowed_values, the question "
            "may refer to a synonym; pick the closest canonical value or "
            "ask the user to clarify."
        )
        overrides["aggregation_hints"] = (
            "Prefer fields marked **good_for_aggregation** for SUM / AVG, "
            "**good_for_grouping** for GROUP BY, and "
            "**good_for_filtering** for WHERE. When multiple numeric "
            "fields could serve an aggregate, the good_for_aggregation "
            "flag is the tie-breaker."
        )

    if resource_description and resource_description.strip():
        overrides["domain_guidance"] = (
            "\n\n## Domain Guidance\n\n" + resource_description.strip()
        )

    return SqlPromptContext(**overrides)
