#!/bin/bash

# Common utility to print UiPath output file
# Usage: source /app/testcases/common/print_output.sh

print_uipath_output() {
    echo "Printing output file..."
    if [ -f "__uipath/output.json" ]; then
        echo "=== OUTPUT FILE CONTENT ==="
        cat __uipath/output.json
        echo "=== END OUTPUT FILE CONTENT ==="
    else
        echo "ERROR: __uipath/output.json not found!"
        echo "Checking directory contents:"
        ls -la
        if [ -d "__uipath" ]; then
            echo "Contents of __uipath directory:"
            ls -la __uipath/
        else
            echo "__uipath directory does not exist!"
        fi
    fi
}
