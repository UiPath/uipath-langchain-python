# Data Fabric agent

A minimal LangGraph agent that queries a Data Fabric entity with
`create_datafabric_tool` with direct entity references.

Replace the entity ID, name, and folder key in `graph.py` with your entity values.

Pass the coded agent's outer system prompt to both `create_agent` and
`create_datafabric_tool`. The latter forwards it to the tool's inner
SQL-generation graph so the same agent instructions apply at both levels.

## Usage

```bash
uv sync
uip codedagent run agent '{"messages":[{"role":"user","content":"List the names."}]}'
```
