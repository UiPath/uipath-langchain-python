"""Assert serde.allowed_msgpack_modules silences langgraph's unregistered-type warning.

The sample's langgraph.json declares `Score` in `serde.allowed_msgpack_modules`.
With the fix, the runtime constructs a strict-mode JsonPlusSerializer that
unions the user's list with the SDK interrupt models, so no warning fires.
"""

import os
import sys

LOG_PATH = "local_run_output.log"
WARNINGS = (
    "Deserializing unregistered type",
    "Blocked deserialization",
)


def main() -> int:
    if not os.path.isfile(LOG_PATH):
        print(f"ERROR: {LOG_PATH} not found")
        return 1

    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        log = f.read()

    found = [w for w in WARNINGS if w in log]
    if found:
        print(f"FAIL: serde warning(s) appeared in log: {found}")
        for line in log.splitlines():
            if any(w in line for w in WARNINGS):
                print(f"  > {line}")
        return 1

    print("OK: no langgraph serde warnings in run output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
