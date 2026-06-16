# SETUP.MD

This file documents how to provision a clean development environment for `uipath-langchain`, run the build, execute the tests, and validate a sample code change end-to-end. It is intended both as a quick reference for human contributors and as a structured guide for automated environment-setup tooling.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 0.5+

### Supported platforms

`uv` is shell- and OS-agnostic, so the commands below run unchanged on every supported platform:

- [x] Linux
- [x] Windows
- [x] macOS

## Environment Variables

None required for environment setup, build, or unit tests. The suite under the `Test` section runs fully offline and requires no external authentication.

> **All commands below must be run from the repository root.** The `uv` invocations resolve `pyproject.toml`, `src/`, and `tests/` relative to the current working directory. The first line of `## Setup` enforces this by `cd`-ing to the git root.

## Setup

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -m pip install --upgrade uv
uv sync --all-extras
```

## Verify Setup

```bash
uv --version
uv run python --version
uv run python -c "import uipath_langchain; print('uipath_langchain ok')"
```

## Build

N/A

## Test

```bash
uv run pytest
```

## Sample Code Change

### The change

Add a new `get_execution_job_id` helper to `src/uipath_langchain/_utils/_environment.py`, immediately after the existing `get_default_timeout` function. The helper is the symmetric complement of the existing `get_execution_folder_path` — it reads the `UIPATH_JOB_KEY` env var at runtime:

```python
def get_execution_job_id() -> str | None:
    """Reads the agent's executing job key from the runtime environment."""
    return os.environ.get("UIPATH_JOB_KEY")
```

Then create `tests/utils/test_environment_job_id.py` with two pytest tests:

```python
import pytest

from uipath_langchain._utils._environment import get_execution_job_id


def test_get_execution_job_id_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UIPATH_JOB_KEY", raising=False)
    assert get_execution_job_id() is None


def test_get_execution_job_id_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UIPATH_JOB_KEY", "job-abc-123")
    assert get_execution_job_id() == "job-abc-123"
```

### Verification

```bash
uv run pytest tests/utils/test_environment_job_id.py -v
```

## Test with a real UiPath Coded LangGraph Agent

> This section is for human contributors who want to validate changes end-to-end against the real cloud platform. It is **not executed by the Agentic Inner Loop validation pipeline** — that pipeline only runs the sections above (Setup → Verify → Build → Test → Sample Code Change).

The unit tests above are necessary but not sufficient — they don't exercise the package end-to-end. The flow below validates changes against a live runtime:

1. Apply the code changes locally.
2. Run the unit tests (see the `Sample Code Change` section above).
3. Scaffold a coded UiPath LangGraph agent that exercises the changed code path.
4. In the downstream project's `pyproject.toml`, add this local library as an editable dependency:

   ```toml
   [tool.uv.sources]
   uipath-langchain = { path = "../path/to/uipath-langchain-python", editable = true }
   ```

5. Exercise the new behavior end-to-end:

   ```bash
   uv run uipath run <agent-name> --input '{...}'
   ```

6. (Optional) Open a PR and apply the `build:dev` label — this publishes the development version to Test PyPI.
7. The PR description is updated automatically with instructions for pointing the downstream agent at the Test PyPI dev version.
8. Validate the new behavior against the real platform — use either or both of the deploy targets below (Studio Web and Orchestrator are not mutually exclusive):
   - **Studio Web**: export the `UIPATH_PROJECT_ID` environment variable pointing to an existing Coded Agent project in your solution, then run [`uipath push`](https://uipath.github.io/uipath-python/cli/#push) to push the dev version to that project. Open it in Studio Web and exercise the changed code path.
   - **Orchestrator**: run [`uipath deploy`](https://uipath.github.io/uipath-python/cli/#deploy) to deploy the dev version as a package, then start a job in Orchestrator and exercise the changed code path.
9. Once validation is done, close the dev PR — these PRs are not meant to be merged; their only purpose was to publish a Test PyPI build for end-to-end validation.
