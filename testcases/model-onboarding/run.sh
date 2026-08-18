#!/bin/bash
set -e

echo "Syncing dependencies..."
uv sync

echo "Authenticating with UiPath..."
# `uipath auth --scope` defaults to 'OR.Execution' alone, which cannot create
# attachments (POST /odata/Attachments -> 403) even though the external app
# grants far more. Request what the agent actually needs.
uv run uipath auth --client-id="$CLIENT_ID" --client-secret="$CLIENT_SECRET" --base-url="$BASE_URL" \
  --scope="OR.Execution OR.Jobs OR.Administration OR.Folders"

# The hosted LLM-as-judge guardrail lives on `agentsruntime_`, which no
# client-credentials app can reach: the OAuth resource catalog has no
# `agentsruntime` entry, so the S2S token is rejected with 401. A PAT carries a
# user identity and is accepted. Swap it in when one is supplied.
#
# The PAT then becomes the ambient token for EVERYTHING below — init, pack,
# and both runs, not just the judge probe — so supply a dedicated
# service-user PAT with minimal scope and a short expiry, never a personal
# one.
#
# This MUST be written into .env rather than exported: the CLI loads .env with
# `override=True` (uipath/_cli/__init__.py), so a process env var loses to
# whatever `uipath auth` just wrote, and the PAT would be silently ignored.
if [ -n "$UIPATH_PAT" ]; then
  echo "Overriding the S2S token with the supplied PAT..."
  uv run python - <<'PY'
import os
from pathlib import Path

env_path = Path(".env")
lines = env_path.read_text().splitlines() if env_path.exists() else []
kept = [ln for ln in lines if not ln.startswith("UIPATH_ACCESS_TOKEN=")]
kept.append(f"UIPATH_ACCESS_TOKEN={os.environ['UIPATH_PAT']}")
env_path.write_text("\n".join(kept) + "\n")
print("UIPATH_ACCESS_TOKEN replaced with the PAT in .env")
PY
fi

echo "Initializing the project..."
uv run uipath init

echo "Packing agent..."
uv run uipath pack

echo "Running agent..."
uv run uipath run agent --file input.json

echo "Running agent again with empty UIPATH_JOB_KEY..."
export UIPATH_JOB_KEY=""
uv run uipath run agent --trace-file .uipath/traces.jsonl --file input.json >> local_run_output.log
