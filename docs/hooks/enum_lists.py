"""MkDocs build hook: expand <!-- ENUM dotted.path.ClassName --> into a markdown table."""

import importlib
import re

PLACEHOLDER = re.compile(r"<!-- ENUM ([\w.]+) -->")


def on_page_markdown(markdown, **kwargs):
    def replace(match):
        dotted = match.group(1)
        module_path, _, cls_name = dotted.rpartition(".")
        enum_cls = getattr(importlib.import_module(module_path), cls_name)
        rows = "\n".join(f"| `{e.name}` |" for e in enum_cls)
        return f"| Value |\n|---|\n{rows}"

    return PLACEHOLDER.sub(replace, markdown)
