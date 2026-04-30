# checkpoint-allowlist

Shows how to silence langgraph's `Deserializing unregistered type` warning by declaring custom types in `langgraph.json`'s `serde.allowed_msgpack_modules` block.

Graph: `START -> evaluate -> finalize -> END`. State carries a custom Pydantic `Score` value, which langgraph reconstructs on every checkpoint round-trip.

## Run

```bash
uv sync
uv run uipath init
uv run uipath run agent --file input.json
```

Expect: agent completes, no `Deserializing unregistered type ...Score` warning. Remove the `serde` block from `langgraph.json` to see the warning return.
