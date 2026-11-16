"""Write the key of an Orchestrator folder to a file, or nothing if it can't be resolved.

The key is never printed: it goes to a file only the current user can read.
"""

import os
import sys
from pathlib import Path

from uipath.platform import UiPath


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            os.environ[key.strip()] = value.strip().strip("'\"")


def main() -> None:
    folder_path, out_file = sys.argv[1], Path(sys.argv[2])
    load_dotenv()
    try:
        key = UiPath().folders.retrieve_key(folder_path=folder_path)
    except Exception as exc:
        print(f"Could not resolve folder {folder_path!r}: {exc}", file=sys.stderr)
        return
    if key:
        fd = os.open(out_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(key)
    else:
        print(f"Folder {folder_path!r} not found", file=sys.stderr)


if __name__ == "__main__":
    main()
