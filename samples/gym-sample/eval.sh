AGENT_NAME=$1
EVAL_SET_NAME=${2:-$1}

mkdir -p results
rm -f __uipath/state.db
uipath eval $AGENT_NAME evals/eval-sets/evaluation-set-$EVAL_SET_NAME.json --no-report --output-file results/eval-results-$EVAL_SET_NAME.json
