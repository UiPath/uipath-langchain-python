# EoG (Explanations over Graphs) Agent Package

## What This Is

A new agent type alongside ReAct for investigative, multi-hop reasoning over ontology-backed virtual knowledge graphs. Based on [arXiv:2601.17915](https://arxiv.org/abs/2601.17915). The agent traverses an ontology's entity-relationship graph deterministically, invokes governed functions at each entity, and uses belief propagation to converge on a minimal explanatory frontier.

## Status: POC — Not Production-Ready

This package is a proof-of-concept. Several components have known limitations that will break in production use. Read the caveats below before making changes.

## Module Layout

| File | What it does | Caveats |
|---|---|---|
| `graph_topology.py` | Parses OWL (.ofn) + YARRRML into an in-memory graph with adjacency | **POC only.** Uses regex to parse OWL Functional Notation — breaks on complex OWL (nested axioms, imports, multi-line). YARRRML parser assumes exact 2-space indent. Production should fetch the resolved graph from a server endpoint. |
| `ontology_client.py` | Async REST client for ontology-runtime | `fetch_graph()` downloads schema.ofn + mapping.yarrrml.yml and parses client-side. Production should call a single discovery endpoint returning JSON. |
| `nodes.py` | 7 LangGraph nodes implementing the EoG algorithm | Function dispatch uses `_bind_params()` which matches param names to entity key properties from YARRRML. Falls back to naming conventions. Entity ID → type resolution uses hardcoded prefix map (`_ID_PREFIXES`). |
| `agent.py` | `create_eog_agent()` graph builder | Returns uncompiled `StateGraph`. Caller compiles with `.compile()`. |
| `types.py` | State types: Belief, LedgerEntry, EoGState, InvestigationConfig | Labels are free strings, not enum — each investigation supplies its own vocabulary. |

## Critical Things to Know

### 1. Graph topology comes from client-side parsing (fragile)

`fetch_graph()` downloads two artifacts and parses them:
- `schema.ofn` → entity types (nodes), object properties (edges), data properties
- `mapping.yarrrml.yml` → entity key property names (e.g., `ToleranceException → exceptionId`)

Both parsers are regex/line-based. They work for the S2P POC's clean, simple artifacts. They WILL break on:
- OWL with nested class expressions, imports, or multi-line axioms
- YARRRML with different indentation or structure

**The right fix:** Add `GET /ontology/{name}/graph` to the ontology-runtime that returns `OntologySnapshot` as JSON. The `OntologyGraph` dataclass stays; only the fetching changes.

### 2. Entity ID → entity type uses prefix conventions

`entity_for_id("INV-2004")` returns `"Invoice"` by matching against `_ID_PREFIXES = {"INV-": "Invoice", ...}`. This is hardcoded for S2P. Other ontologies need their own prefix map or a fundamentally different approach (server-side resolution).

### 3. Function param binding depends on YARRRML key properties

Functions like `exceptionContext(exceptionId)` are bound to entity instances via the key property name from YARRRML (`ToleranceException → exceptionId`). Without this mapping, `_bind_params` can't match `exceptionId` to `ToleranceException` — it would expect `toleranceExceptionId`.

If a function has a required param that can't be bound, the function is skipped for that entity. This is intentional — it means the function is investigation-scoped (e.g., `spendByCategory(period)`) not entity-scoped.

### 4. Propagation follows graph edges, NOT LLM suggestions

When an entity's label changes, the `propagate_node` broadcasts to all graph neighbors (entities connected by OWL object properties). The LLM does NOT decide who to propagate to — that's the whole point of EoG vs ReAct.

Propagation only happens when the label actually changed. Neighbors are only re-activated if their flip count is below `max_flips` (damping).

### 5. Frontier computes irreducibility

The frontier is NOT just "non-Defer beliefs." It removes entities that are explained by another entity: if Source A has an explanatory edge to DerivedEffect B, then B is removed from the frontier (it's explained by A).

### 6. Namespace must be `ont#`

All OWL/SHACL/YARRRML/FnO artifacts must use `https://ontology.uipath.com/ont#`. The ontology-runtime hardcodes this in `Vocabulary.ONT`. Custom namespaces silently break Ontop reformulation.

### 7. Ontop does not support transitive properties

`TransitiveObjectProperty` is accepted but has no effect. SPARQL property paths (`+`, `*`) are rejected. Hierarchy traversal (Supplier parent, Commodity taxonomy) needs materialized paths or precomputed functions.

## Running Tests

```bash
uv run pytest tests/agent/eog/ -v    # 60 tests
uv run ruff check src/uipath_langchain/agent/eog/
```

## Running the Live Agent

Requires the ontology-runtime local stack (see `eog_architecture_summary.md` in memory or the `examples/s2p-ontology/` directory).

```bash
cd examples/s2p-eog-agent
source .venv/bin/activate
uip codedagent run agent '{"question": "Investigate open tolerance exceptions"}'
```
