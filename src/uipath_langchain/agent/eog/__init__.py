"""EoG (Explanations over Graphs) agent package.

Stubs for the EoG agent API. The real implementation is being built in
parallel (WU2). These placeholders let downstream code import and type-check
against the expected surface.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field


class OntologyClient:
    """HTTP client for the UiPath Ontology Runtime.

    Args:
        base_url: Root URL of the ontology-runtime service.
        account: UiPath account (logical tenant group).
        tenant: UiPath tenant within the account.
    """

    def __init__(
        self,
        base_url: str,
        account: str,
        tenant: str,
    ) -> None:
        self.base_url = base_url
        self.account = account
        self.tenant = tenant


class InvestigationConfig(BaseModel):
    """Configuration that governs a single EoG investigation run."""

    label_vocabulary: list[str] = Field(
        default_factory=list,
        description="Allowed belief labels the agent can assign.",
    )
    seed_entities: list[str] = Field(
        default_factory=list,
        description="Entity types or IRIs that seed the investigation.",
    )
    max_steps: int = Field(
        default=30,
        description="Maximum investigation steps before forced termination.",
    )
    max_flips: int = Field(
        default=3,
        description="Maximum label flips per entity before it is frozen.",
    )
    default_label: str = Field(
        default="Defer",
        description="Label assigned when evidence is insufficient.",
    )
    max_results_per_function: int = Field(
        default=50,
        description="Cap on rows returned by any single ontology query.",
    )


class Belief(BaseModel):
    """A belief about a single entity in the investigation."""

    label: str = Field(description="Current classification label.")
    evidence: str = Field(
        default="", description="Free-text evidence supporting the label."
    )
    flip_count: int = Field(
        default=0, description="Number of times the label has changed."
    )


class EoGState(BaseModel):
    """Minimal state schema for an EoG investigation graph."""

    investigation_config: Optional[InvestigationConfig] = None
    beliefs: dict[str, Belief] = Field(default_factory=dict)
    ledger: list[dict[str, Any]] = Field(default_factory=list)
    frontier: list[str] = Field(default_factory=list)
    steps_taken: int = 0


def create_eog_agent(
    model: Any,
    ontology_client: OntologyClient,
    ontology_name: str,
    *,
    investigation_config: InvestigationConfig,
) -> StateGraph:
    """Build an EoG investigation graph (uncompiled).

    Args:
        model: LangChain chat model used for reasoning.
        ontology_client: Client connected to the ontology-runtime.
        ontology_name: Name of the deployed ontology to query.
        investigation_config: Parameters controlling the investigation.

    Returns:
        An uncompiled ``StateGraph`` ready for ``.compile()``.
    """
    raise NotImplementedError(
        "create_eog_agent is a stub. "
        "Install the real eog package (WU2) to use this function."
    )


__all__ = [
    "Belief",
    "EoGState",
    "InvestigationConfig",
    "OntologyClient",
    "create_eog_agent",
]
