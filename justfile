set quiet

default: lint format

lint:
    ruff check .
    python scripts/lint_httpx_client.py

format:
    ruff format --check .
    ruff check --fix

validate: lint format

build:
    uv build

install:
    uv sync --all-extras
