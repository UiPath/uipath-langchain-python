"""State types for the EoG (Explanations over Graphs) agent.

The EoG agent traverses an ontology lazily — no full graph fetch.
Function definitions (with ``touches``, ``outputs``, ``params``) are
the navigation contract: ``touches`` = adjacency, ``outputs`` = what
evidence comes back, ``params`` = what values you need to get there.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

from uipath_langchain.agent.react.reducers import merge_dicts


class Belief(BaseModel):
    """A belief about a record's role in the investigation."""

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
    """An edge connecting two records in the explanation graph."""

    source: str
    target: str
    relationship: str
    evidence: str = ""


class FunctionSpec(BaseModel):
    """A self-describing function definition from the ontology runtime.

    This is the navigation + evidence contract:
    - ``touches`` = which entity types this function reads (adjacency)
    - ``outputs`` = typed columns the function returns (evidence schema)
    - ``params`` = what the function needs as input
    - ``description`` = natural language contract for the LLM

    Accepts both the flat format (``params``/``outputs``/``touches``) and
    the wire format (``input_schema``/``output_schema`` JSON Schema objects).
    """

    name: str
    label: str = ""
    description: str = ""
    language: str | None = None
    statement: str | None = None
    params: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    touches: list[str] = Field(default_factory=list)
    # Wire format fields (JSON Schema)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    def model_post_init(self, __context: Any) -> None:
        """Normalize wire format to flat lists."""
        if self.input_schema and not self.params:
            self.params = _schema_to_params(self.input_schema)
        if self.output_schema and not self.outputs:
            self.outputs = _schema_to_params(self.output_schema)
        if not self.touches and self.statement:
            self.touches = _infer_touches(self.statement)

    @property
    def required_params(self) -> list[str]:
        """Names of required parameters."""
        return [p["name"] for p in self.params if p.get("required", False)]

    @property
    def param_names(self) -> set[str]:
        """All parameter names."""
        return {p["name"] for p in self.params}

    @property
    def output_names(self) -> set[str]:
        """All output column names."""
        return {o["name"] for o in self.outputs}


def _schema_to_params(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert JSON Schema ``input_schema``/``output_schema`` to flat param list."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    return [
        {"name": name, "type": meta.get("type", "string"), "required": name in required}
        for name, meta in props.items()
    ]


def _infer_touches(statement: str) -> list[str]:
    """Infer entity types from SPARQL ``ont:EntityName`` references."""
    import re
    # Match "ont:CapitalizedWord" that appear as class references (after "a ont:" or as property prefix)
    matches = re.findall(r"ont:([A-Z][a-zA-Z]+)(?:\.|[ ;])", statement)
    return sorted(set(matches))


class InvestigationConfig(BaseModel):
    """Configuration that makes EoG reusable across domains.

    Args:
        label_vocabulary: Allowed labels, e.g. ["Source", "DerivedEffect"].
        seed_records: Starting record IDs for the traversal.
        max_steps: Hard cap on record visits.
        max_flips: Damping threshold — records that flip more than this
            are no longer re-activated by neighbours.
        default_label: Label assigned before investigation.
        max_results_per_function: Upper bound on rows per function call.
        max_tokens: Token budget — stop when cumulative LLM tokens exceed this.
            0 means no token limit (step budget only).
        convergence_window: Stop early when this many consecutive visits
            produce no belief change. 0 disables convergence detection.
    """

    label_vocabulary: list[str]
    seed_records: list[str] = Field(default_factory=list)
    max_steps: int = 50
    max_flips: int = 3
    default_label: str = "Defer"
    max_results_per_function: int = 50
    max_tokens: int = 0
    convergence_window: int = 5


class EoGState(BaseModel):
    """LangGraph state for the EoG investigation agent.

    No full ontology graph in state. Instead:
    - ``function_cache``: entity_type → list of FunctionSpec dicts (lazy, per-type)
    - ``discovered_records``: entity_id → entity_type (grows as functions return results)
    """

    active_set: list[str] = Field(default_factory=list)
    current_record: str = ""
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
    investigation_config: InvestigationConfig | None = None

    # Lazy graph state — no upfront fetch
    function_cache: Annotated[
        dict[str, list[dict[str, Any]]], merge_dicts
    ] = Field(default_factory=dict)
    discovered_records: Annotated[dict[str, str], merge_dicts] = Field(
        default_factory=dict
    )

    # Budget tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    consecutive_no_change: int = 0
