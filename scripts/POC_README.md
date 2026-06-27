# Data Fabric Ontology Write POC

Demonstrates an OWL ontology driving native Data Fabric **writes** through the
agent tooling: the ontology compiles into a structured intermediate
representation, activates in the write handler, governs which entities/operations
are allowed, and the resulting writes persist to Data Fabric.

Hero case: a contact-center **refund** agent over 5 entities — `Customer`
(read-only), `Contact`, `PurchaseOrder`, `CustomerRisk`, `RefundRequest`.

## Components

| File | Role |
|------|------|
| `src/.../datafabric_tool/ontology_compiler.py` | OWL Turtle → `CompiledOntology` (entity_access, measure/state/reference fields, HITL, relationships) |
| `src/.../datafabric_tool/compiled_ontology.py` | the IR model |
| `src/.../datafabric_tool/datafabric_tool.py` | write tool + handler; fetches+compiles ontology, maps entity name→id for CRUD |
| `src/.../datafabric_tool/write_validation.py` | writability + mutation-intent validation (ontology-constrained) |
| `src/.../datafabric_tool/datafabric_prompt_builder.py` | read schema; retains the primary key for writable entities |
| `scripts/poc_refund_setup.sh` | create + seed staging entities, emit ontology + entity-set + ids |
| `scripts/poc_refund_drive.py` | drive the real write handler with the ontology active, verify by read-back |
| `scripts/poc_refund_teardown.sh` | delete the POC entities |
| `scripts/run_agent_with_ontology.py` | full LLM-in-the-loop variant (see "Known gap") |

## Prerequisites

```bash
# 1. CLI auth to the target tenant (entity create/seed/verify)
uip login

# 2. SDK env vars — the Python SDK reads these (separate from the CLI's auth).
#    Source the access token from a logged-in session; do NOT hardcode it.
export UIPATH_ACCESS_TOKEN="$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.uipath/.auth.json')))['access_token'])")"
export UIPATH_URL="https://<host>/<org>/<tenant>"
export UIPATH_ORGANIZATION_ID="<org-guid>"
export UIPATH_TENANT_ID="<tenant-guid>"
```

The access token is short-lived (~1h); re-export after re-login.

## Run

### A. Ontology compiles + activates (offline, no staging)

```bash
uv run python scripts/run_agent_with_ontology.py \
  --ontology ../../../df-agent-os/roadmap/p1-owl-write-extension.ttl \
  --entity-set scripts/sample_refund_entity_set.json \
  --prompt x --dry-run
```

Prints the extracted ontology facts without any network call.

### B. Ontology governs writes that persist (live staging) — the working POC

```bash
bash scripts/poc_refund_setup.sh ./poc_out        # create + seed
uv run python scripts/poc_refund_drive.py ./poc_out   # drive writes, verify
bash scripts/poc_refund_teardown.sh ./poc_out     # clean up
```

`poc_refund_drive.py` prints `ontology ACTIVE`, runs insert RefundRequest +
update Order/CustomerRisk/Contact through the real handler, and verifies all
four mutations by read-back.

### C. Full LLM agent picks the tools (live)

```bash
set -a; source ./poc_out/refund_ids.env; set +a
uv run python scripts/run_agent_with_ontology.py \
  --ontology ./poc_out/refund_ontology.ttl \
  --entity-set ./poc_out/refund_entity_set.json \
  --system-prompt scripts/sample_refund_sop.txt \
  --model gpt-4.1-2025-04-14 --agenthub-config agentsplayground \
  --prompt "Process the refund for contact ${CONTACT_ID}. Order id ${ORDER_ID}, CustomerRisk id ${RISK_ID}, Customer id ${CUSTOMER_ID}. ..."
```

The LLM reads, decides, and emits ontology-correct write calls (insert on
RefundRequest, update on the writable entities, never on read-only Customer).

## Known gap

In path **C**, the standalone `create_agent` harness terminates on control-flow
tools and the gateway returns tool calls in the OpenAI Responses format; the
terminal write batch is *planned* but not auto-executed by this harness. That is
agent-runtime plumbing, not the ontology or the write tool — path **B** confirms
the writes themselves land. The production `uipath_agents` runtime drives the
tool-execution loop and is the place to validate path C end-to-end.

## Notes

- Status fields are plain STRING (the `choice-set-values` endpoint was
  unreliable on staging); the ontology still models `OrderStatus` as a
  `StateField`. Swap to ChoiceSet fields when the endpoint is stable.
- The seeded entities are not FK-linked (simplified scenario); pass the record
  ids explicitly in path C rather than relying on relationship discovery.
