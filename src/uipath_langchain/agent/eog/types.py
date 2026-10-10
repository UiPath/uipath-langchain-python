"""State types for the EoG (Explanations over Graphs) agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

from uipath_langchain.agent.react.reducers import merge_dicts


class Belief(BaseModel):
    """A belief about an entity's label with supporting evidence.

    Labels are free strings drawn from InvestigationConfig.label_vocabulary.
    """

    label: str
    evidence: str = ""
    flip_count: int = 0


class LedgerEntry(BaseModel):
    """Append-only record of a belief update."""

    timestamp: float
    entity_id: str
    old_label: str | None
    new_label: str
    evidence: str
    function_used: str | None = None


class ExplanatoryEdge(BaseModel):
    """An edge in the explanatory graph connecting two entities."""

    source: str
    target: str
    relationship: str
    evidence: str = ""


class InvestigationConfig(BaseModel):
    """Configuration that makes EoG reusable across domains.

    Args:
        label_vocabulary: Allowed labels, e.g. ["Source", "DerivedEffect"].
        seed_entities: Starting entity IDs for the BFS traversal.
        max_steps: Budget cap on the number of entity visits.
        max_flips: Damping threshold -- entities that flip more than this
            are no longer re-activated by neighbours.
        default_label: Label assigned to entities before investigation.
        max_results_per_function: Upper bound on rows returned by each
            ontology function invocation.
    """

    label_vocabulary: list[str]
    seed_entities: list[str] = Field(default_factory=list)
    max_steps: int = 50
    max_flips: int = 3
    default_label: str = "Defer"
    max_results_per_function: int = 50


class EoGState(BaseModel):
    """LangGraph state for the EoG investigation agent."""

    active_set: list[str] = Field(default_factory=list)
    current_entity: str = ""
    beliefs: Annotated[dict[str, Belief], merge_dicts] = Field(
        default_factory=dict
    )
    inbox: Annotated[dict[str, list[dict[str, Any]]], merge_dicts] = Field(
        default_factory=dict
    )
    ledger: Annotated[list[LedgerEntry], operator.add] = Field(
        default_factory=list
    )
    explanatory_edges: Annotated[list[ExplanatoryEdge], operator.add] = Field(
        default_factory=list
    )
    context_packet: dict[str, Any] = Field(default_factory=dict)
    policy_result: dict[str, Any] = Field(default_factory=dict)
    steps_taken: int = 0
    frontier: list[dict[str, Any]] = Field(default_factory=list)
    ontology_graph: dict[str, Any] = Field(default_factory=dict)
    investigation_config: InvestigationConfig | None = None
