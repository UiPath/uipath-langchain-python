"""Upload files as Orchestrator attachments and print the agent input for each.

Usage (from this folder):
    uv run python upload_attachment.py <file> [<file> ...] [--refresh-token]

Uses the credentials in .env. --refresh-token first runs `uip login refresh`
and writes the new access token into .env.
"""

import argparse
import asyncio
import json
import mimetypes
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).with_name(".env")

MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def refresh_token() -> None:
    output = subprocess.run(
        ["uip", "login", "refresh", "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    token = json.loads(output)["Data"]["AccessToken"]
    text = ENV_FILE.read_text()
    text = re.sub(
        r"^UIPATH_ACCESS_TOKEN=.*$", f"UIPATH_ACCESS_TOKEN={token}", text, flags=re.M
    )
    ENV_FILE.write_text(text)
    print("Access token refreshed in .env")


def mime_type(path: Path) -> str:
    return (
        MIME_TYPES.get(path.suffix.lower())
        or mimetypes.guess_type(path.name)[0]
        or "application/octet-stream"
    )


async def upload(paths: list[Path]) -> None:
    from uipath.platform import UiPath

    uipath = UiPath()
    for path in paths:
        attachment_id = await uipath.attachments.upload_async(
            name=path.name, source_path=str(path)
        )
        agent_input = {
            "file": {
                "ID": str(attachment_id),
                "FullName": path.name,
                "MimeType": mime_type(path),
            },
            "recruiterNotes": "",
        }
        print(f"\n{path.name}  ->  {attachment_id}")
        print(f"uv run uipath run agent '{json.dumps(agent_input)}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--refresh-token", action="store_true")
    args = parser.parse_args()

    missing = [str(path) for path in args.files if not path.is_file()]
    if missing:
        parser.error(f"not a file: {', '.join(missing)}")
    if args.refresh_token:
        refresh_token()
    load_dotenv(ENV_FILE, override=True)
    asyncio.run(upload(args.files))


if __name__ == "__main__":
    main()
