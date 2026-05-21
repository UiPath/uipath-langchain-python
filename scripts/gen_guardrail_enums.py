"""Regenerates enum value tables in docs/guardrails.md from the installed uipath package.

Run manually: python scripts/gen_guardrail_enums.py
Also runs automatically as a pre-commit hook when docs/guardrails.md or enums.py is staged.
"""

import pathlib
import re
import sys

from uipath_langchain.guardrails.enums import (
    HarmfulContentEntityType,
    IntellectualPropertyEntityType,
    PIIDetectionEntityType,
)

ENUMS = [PIIDetectionEntityType, HarmfulContentEntityType, IntellectualPropertyEntityType]
DOCS = pathlib.Path(__file__).parent.parent / "docs" / "guardrails.md"


def make_table(enum_cls):
    rows = "\n".join(f"| `{m.name}` |" for m in enum_cls)
    return f"| Value |\n|---|\n{rows}"


def regenerate(content, enum_cls):
    name = enum_cls.__name__
    table = make_table(enum_cls)
    marker = rf"(<!-- BEGIN ENUM {name} -->).*?(<!-- END ENUM {name} -->)"
    replacement = rf"\1\n{table}\n\2"
    new, n = re.subn(marker, replacement, content, flags=re.DOTALL)
    if n == 0:
        print(f"WARNING: markers for {name} not found in {DOCS}", file=sys.stderr)
    return new


def main():
    text = DOCS.read_text()
    for cls in ENUMS:
        text = regenerate(text, cls)
    DOCS.write_text(text)
    print(f"Regenerated enum tables in {DOCS}")


if __name__ == "__main__":
    main()
