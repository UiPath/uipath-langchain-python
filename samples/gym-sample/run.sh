#!/bin/bash

# Usage:
#   ./run.sh calculator                              # Uses test_input.json
#   ./run.sh calculator '{"expression": "2 + 2"}'    # Uses direct JSON input

if [ -z "$2" ]; then
    # No second argument - use test_input.json file
    uipath run $1 -f src/gym_sample/$1/test_input.json
else
    # Second argument provided - use it directly
    uipath run $1 "$2"
fi
