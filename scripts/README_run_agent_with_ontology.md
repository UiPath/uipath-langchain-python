# run_agent_with_ontology.py

A standalone CLI helper that loads an OWL ontology, injects it into the Data
Fabric **write** tool's fetch path, and runs a coded ReAct agent against a Data
Fabric entity set.

This is a developer helper, not part of the shipped package. No tests cover it.

## What it does

1. **Compiles + prints the ontology up front.** It runs
   `compile_ontology()` on your `.ttl` and prints the extracted facts —
   `entity_access`, `hitl_operations`, `state_fields`, `reference_fields`,
   `measure_fields`, `entity_relationships`. This validates the `.ttl` and
   shows exactly what the ontology contributes before any agent work. This step
   needs **no network**.
2. **Bridges a not-yet-shipped platform method** (see below).
3. Builds the Data Fabric read + write tools, force-initializes the write
   handler, prints the compiled ontology actually in use and the generated
   write tool description, then (unless `--dry-run`) runs the agent.

## The monkeypatch bridge — and why it exists

The runtime's `DataFabricWriteHandler._maybe_compile_ontology()` fetches the
ontology by calling:

```python
entities_service.get_ontology_file_async("owl")
```

**The platform does not yet expose that method.** Verified: the attribute is
absent on
`uipath.platform.entities._entities_service.EntitiesService`. When it is
missing, the handler degrades to the metadata-only write path
(`_compiled_ontology` stays `None`).

This CLI closes that gap by monkeypatching the method onto the class **before**
the handler runs:

```
uipath.platform.entities._entities_service.EntitiesService.get_ontology_file_async
```

The injected async method returns the text of your `--ontology` file for the
`"owl"` file type. The handler's own `_maybe_compile_ontology` then discovers
it via `getattr`, compiles it, and uses it in write validation and the write
tool description — exactly as it will behave once the platform ships the real
method.

A class-level patch is used (not an instance patch) because the handler
constructs `UiPath()` internally and resolves the `EntitiesService` lazily, so
the instance is never reachable from the CLI. If the class patch ever fails,
the script falls back to setting `handler._compiled_ontology` directly and
rebuilding the description.

After initialization the script prints either:

- `ontology ACTIVE` — the handler compiled and is using the ontology, or
- `ontology INACTIVE (fell back to metadata-only)` — it did not.

## Run it offline (dry-run)

The ontology compilation + fact printing needs no network. Building tools /
resolving entities **does** need UiPath auth + network; in `--dry-run` that
failure is caught and the script still prints the standalone ontology facts and
exits `0`.

```bash
uv run python scripts/run_agent_with_ontology.py \
    --ontology /Users/harshit/DF-Agents-2/df-agent-os/roadmap/p1-owl-write-extension.ttl \
    --entity-set scripts/sample_refund_entity_set.json \
    --prompt "test" --dry-run
```

## Run it for real (against staging)

1. `uip login` (sets `UIPATH_ACCESS_TOKEN`, `UIPATH_URL`,
   `UIPATH_TENANT_ID`, `UIPATH_ORGANIZATION_ID`).
2. Edit `scripts/sample_refund_entity_set.json` and replace the **placeholder
   fake UUIDs** (`id`, `folderId`, `referenceKey`) with the real ids for your
   tenant's entities. The shipped values are clearly fake and will not resolve.
3. Run without `--dry-run`:

```bash
uv run python scripts/run_agent_with_ontology.py \
    --ontology /path/to/ontology.ttl \
    --entity-set scripts/sample_refund_entity_set.json \
    --prompt "Process the refund for contact Jane Doe on order PO-1042" \
    --model anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --system-prompt scripts/sample_refund_sop.txt
```

## Options

| Flag | Required | Description |
|------|----------|-------------|
| `--ontology` | yes | Path to the OWL 2 QL Turtle `.ttl` file. |
| `--entity-set` | yes | JSON list of `DataFabricEntityItem` dicts (`id`, `name`, `folderId`, `referenceKey`, `description`). |
| `--prompt` | yes | The user prompt for the agent. |
| `--model` | no | UiPath-gateway model name. Default: `anthropic.claude-sonnet-4-5-20250929-v1:0`. |
| `--system-prompt` | no | Path to a system-prompt/SOP `.txt`. Generic default when omitted. |
| `--resource-name` | no | Name for the Data Fabric context resource. Default: `datafabric`. |
| `--dry-run` | no | Do not call the LLM; build tools, inject the ontology, print facts + write tool description, exit. Degrades gracefully offline. |

## Sample files

- `sample_refund_entity_set.json` — refund hero-case entities (Customer,
  Contact, Order/PurchaseOrder, CustomerRisk, RefundRequest) with **placeholder
  fake ids**. Fill in real tenant ids to run against staging.
- `sample_refund_sop.txt` — the refund SOP (RFC §4.3) as a system prompt.
