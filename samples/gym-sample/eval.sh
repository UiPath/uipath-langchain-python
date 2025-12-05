#!/bin/bash

# Usage:
#   ./eval.sh calculator                    # Runs calculator agent with evaluation-set-calculator.json
#   ./eval.sh loan                          # Runs loan agent with evaluation-set-loan.json
#   ./eval.sh calculator custom-eval-set    # Runs calculator agent with evaluation-set-custom-eval-set.json

# First argument: agent name (e.g., "calculator", "loan")
AGENT_NAME=$1

# Second argument: eval set name (defaults to agent name if not provided)
EVAL_SET_NAME=${2:-$1}

NUM_WORKERS=${3:-1}

# Create results directory if it doesn't exist
mkdir -p results

# Clean up previous state database to ensure fresh evaluation run
rm -f __uipath/state.db

# Run UiPath evaluation
# - Uses the agent specified by $AGENT_NAME
# - Loads eval set from evaluations/eval-sets/evaluation-set-$EVAL_SET_NAME.json
# - Skips sending report to UiPath platform
# - Outputs results to results/eval-results-$EVAL_SET_NAME.json
uipath eval $AGENT_NAME evaluations/eval-sets/evaluation-set-$EVAL_SET_NAME.json --no-report --output-file results/eval-results-$EVAL_SET_NAME.json --workers $NUM_WORKERS
