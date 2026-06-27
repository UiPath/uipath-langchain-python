#!/bin/bash
# Delete the POC entities created by poc_refund_setup.sh.
# Usage: bash scripts/poc_refund_teardown.sh [OUT_DIR]   (default: ./poc_out)
set -eo pipefail
OUT="${1:-./poc_out}"
set -a; source "$OUT/refund_ids.env"; set +a
for e in "$E_CUSTOMER" "$E_CONTACT" "$E_ORDER" "$E_RISK" "$E_REFUND"; do
  uip df entities delete "$e" --yes --reason "poc teardown" --output json 2>/dev/null \
    | python3 -c "import json,sys;print(json.load(sys.stdin).get('Code','?'))" 2>/dev/null || echo "skip $e"
done
echo "teardown done"
