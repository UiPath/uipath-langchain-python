# S2P EoG Investigation Agent

Sample agent demonstrating the **EoG (Explanations over Graphs)** pattern on
the Source-to-Pay (S2P) procurement ontology.

The agent connects to a local ontology-runtime, queries the S2P knowledge graph,
and iteratively builds an *explanatory frontier* -- a set of labelled beliefs
that diagnose procurement issues such as tolerance exceptions, maverick spend,
and invoice mismatches.

## Prerequisites

1. **FQS mock** running (provides the SPARQL/query backend):

   ```bash
   cd ../fqs-data && ./fqs_start.sh
   ```

2. **Ontology-runtime** running:

   ```bash
   java -jar ontology-app.jar --spring.profiles.active=local
   ```

3. **S2P ontology deployed**:

   ```bash
   cd ../scripts && ./run-e2e.sh
   ```

4. **Environment variables** -- copy `.env.example` to `.env` and fill in values
   (or export them directly):

   ```bash
   cp .env.example .env
   ```

## Running the test script

```bash
cd examples/s2p-ontology/agent
python test_eog.py
```

## Expected output

The script prints a step-by-step ledger of label assignments, the final belief
map for each investigated entity, and a summary frontier of findings. Example
(abbreviated):

```
S2P EoG Investigation Agent -- Test Run
============================================================
Seeding investigation with: ['ToleranceException', 'Invoice', ...]
------------------------------------------------------------

INVESTIGATION COMPLETE
============================================================
Ledger (12 entries):
  [inv-001] -- -> Source: Invoice amount exceeds PO by 15%
  ...
Final Beliefs:
  [inv-001] Source: Invoice amount exceeds PO by 15%
  [po-042]  DerivedEffect: PO linked to flagged supplier
  ...
Steps taken: 12
```

## LangGraph configuration

`langgraph.json` exposes the compiled graph as `agent` so it can be served by
the LangGraph CLI or deployed to UiPath Cloud via `uipath run`.
