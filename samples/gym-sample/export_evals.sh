#!/bin/bash

# Usage:
#   ./export_evals.sh                    # Export all evaluators
#   ./export_evals.sh --include_not_supported   # Include not supported evaluators
#   ./export_evals.sh --exclude_llm_judge   # Exclude LLM judge evaluators
#   ./export_evals.sh --small_set_size 10   # Export a small set of 10 datapoints

# Export evaluators and eval sets to UiPath eval format
uv run python -m gym_sample.export_evals "$@"

