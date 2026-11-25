# Ontology agent

A minimal LangGraph agent that queries a deployed ontology with
`create_ontology_tool`, addressing the ontology directly by its id.

Replace the ontology id in `graph.py` with the id of an ontology deployed in your
tenant. The tool reads the ontology on its first call and fails that call if the
ontology is not deployed.

Pass the coded agent's outer system prompt to both `create_agent` and
`create_ontology_tool`. The latter forwards it to the tool's inner
SQL-generation graph so the same agent instructions apply at both levels.

## Usage

```bash
uv sync
uip codedagent run agent '{"messages":[{"role":"user","content":"How many invoices are unpaid?"}]}'
```
