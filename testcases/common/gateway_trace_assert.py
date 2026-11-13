"""Check the LLM Gateway's spans of a run, read back from LLMOps.

"required_gateway_spans" in expected_traces.json uses the "required_spans"
format, plus:
- "min" / "max": how many matching spans may exist (default: at least 1);
- "parent": "local" (default) if the span's parent must be in traces.jsonl,
  or "any";
- "parent_name": the name (or list of names) of that local parent span.
"""

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from trace_assert import load_traces, matches_expected

TRACE_ID_FILE = ".uipath/gateway_trace_id"
SPANS_PATH = "/llmopstenant_/api/Traces/v2/spans/otel?traceId={trace_id}&pageSize=1000"


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            os.environ[key.strip()] = value.strip().strip("'\"")


def _span_id(value: Any) -> str:
    return str(value or "").lower().removeprefix("0x")[-16:]


def _normalize(span: dict[str, Any]) -> dict[str, Any]:
    attributes = span.get("Attributes") or {}
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    return {
        "name": span.get("Name"),
        "SpanType": span.get("SpanType"),
        "attributes": attributes,
        "parent": _span_id(span.get("ParentId")),
    }


class _NoAccess(Exception):
    pass


def _skip(reason: str) -> None:
    message = f"LLM Gateway span check skipped: {reason}"
    # Shown as a warning annotation on the GitHub Actions run.
    print(
        f"::warning::{message}"
        if os.environ.get("GITHUB_ACTIONS") == "true"
        else message
    )


def _fetch(trace_id: str) -> list[dict[str, Any]] | None:
    """The trace's spans, or None on an error worth retrying."""
    url = os.environ["UIPATH_URL"].rstrip("/") + SPANS_PATH.format(trace_id=trace_id)
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {os.environ['UIPATH_ACCESS_TOKEN']}",
            "Accept": "application/json",
            # Cloudflare rejects urllib's default User-Agent with a 403.
            "User-Agent": "uipath-langchain-testcases",
        },
    )
    context = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise _NoAccess(f"{exc.code} {exc.reason}") from exc
        print(f"LLMOps returned {exc.code} {exc.reason}")
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"LLMOps request failed: {exc}")
        return None
    assert not data.get("Tokens"), "The LLMOps trace has more than one page of spans"
    return [_normalize(s) for s in data.get("Spans", [])]


def _missing(
    spans: list[dict[str, Any]],
    expectations: list[dict[str, Any]],
    local_names: dict[str, str],
) -> list[str]:
    missing = []
    for expected in expectations:
        criteria = {
            k: v
            for k, v in expected.items()
            if k in ("name", "span_type", "attributes")
        }
        found = [s for s in spans if matches_expected(s, criteria)]
        if expected.get("parent", "local") == "local":
            found = [s for s in found if s["parent"] in local_names]
        if "parent_name" in expected:
            parents = expected["parent_name"]
            parents = [parents] if isinstance(parents, str) else parents
            found = [s for s in found if local_names.get(s["parent"]) in parents]
        low = int(expected.get("min", 0 if "max" in expected else 1))
        high = expected.get("max")
        label = str(expected["name"])
        if "parent_name" in expected:
            label += f" under {expected['parent_name']}"
        if expected.get("attributes"):
            label += f" {json.dumps(expected['attributes'])}"
        ok = len(found) >= low and (high is None or len(found) <= int(high))
        need = f"{low}" if high is None else f"{low}..{high}"
        print(
            f"{'✓' if ok else '✗'} Gateway span {label}: {len(found)} found, need {need}"
        )
        if not ok:
            missing.append(label)
    return missing


def assert_gateway_traces(
    traces_file: str,
    expected_file: str,
    trace_id_file: str = TRACE_ID_FILE,
    timeout: float = 60,
) -> None:
    with open(expected_file, encoding="utf-8") as f:
        expectations = json.load(f).get("required_gateway_spans", [])
    if not expectations:
        return
    if not Path(trace_id_file).is_file():
        _skip(f"no {trace_id_file}")
        return

    _load_dotenv()
    trace_id = Path(trace_id_file).read_text(encoding="utf-8").strip()
    local_names = {
        _span_id(s["context"]["span_id"]): s["name"] for s in load_traces(traces_file)
    }
    print(f"Checking LLM Gateway spans in LLMOps trace {trace_id}...")

    deadline = time.monotonic() + timeout
    # Spans keep arriving, so an upper bound only holds on the last snapshot.
    has_max = any("max" in e for e in expectations)
    spans: list[dict[str, Any]] = []
    missing = [str(e["name"]) for e in expectations]
    while True:
        try:
            fetched = _fetch(trace_id)
        except _NoAccess as exc:
            _skip(f"no access to LLMOps ({exc})")
            return
        if fetched is not None:
            spans = fetched
            missing = _missing(spans, expectations, local_names)
        elif has_max:
            missing = ["max bounds (the last LLMOps fetch failed)"]
        if (not missing and not has_max) or time.monotonic() >= deadline:
            break
        if missing:
            print("Not all Gateway spans are in LLMOps yet, retrying in 10s...")
        else:
            print("Waiting until the deadline to check the max bounds...")
        time.sleep(10)

    if missing:
        names = sorted({str(s["name"]) for s in spans})
        raise AssertionError(
            f"LLM Gateway span checks failed: {', '.join(missing)}. "
            f"Spans in the trace: {', '.join(names) or '(none)'}"
        )
    print(f"✓ All {len(expectations)} expected LLM Gateway spans found!")
