#!/bin/bash

# Script to collect performance counters using perf stat
# Usage: ./collect_nnpa_stats.sh <counter_file> <python_script> [args...]

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <counter_file> <python_script> [args...]"
    echo ""
    echo "Arguments:"
    echo "  counter_file  - File containing performance counter names (one per line)"
    echo "  python_script - Python script to run"
    echo "  args...       - Additional arguments for the Python script"
    echo ""
    echo "Example: $0 counters.txt run-matmul.py -m ./matmul-dynamic.so"
    echo ""
    echo "Counter file format (one counter name per line):"
    echo "  NNPA_4K_PREFETCH"
    echo "  NNPA_COMPLETIONS"
    echo "  NNPA_INVOCATIONS"
    exit 1
fi

# Get the counter file and shift arguments
COUNTER_FILE="$1"
shift

# Check if counter file exists
if [ ! -f "$COUNTER_FILE" ]; then
    echo "Error: Counter file not found: $COUNTER_FILE"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to virtual environment Python
VENV_PYTHON="/home/chentong/workspace/test-env/bin/python"

# Check if virtual environment Python exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment Python not found at $VENV_PYTHON"
    echo "Please ensure the test-env virtual environment exists in $SCRIPT_DIR"
    exit 1
fi

# Read counters from file into array, skipping empty lines and comments
COUNTERS=()
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and lines starting with #
    if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
        # Trim whitespace
        counter=$(echo "$line" | xargs)
        if [ -n "$counter" ]; then
            COUNTERS+=("$counter")
        fi
    fi
done < "$COUNTER_FILE"

# Check if we have any counters
if [ ${#COUNTERS[@]} -eq 0 ]; then
    echo "Error: No valid counters found in $COUNTER_FILE"
    exit 1
fi

# Build the perf stat command with all counters
COUNTER_LIST=""
for counter in "${COUNTERS[@]}"; do
    if [ -z "$COUNTER_LIST" ]; then
        COUNTER_LIST="$counter"
    else
        COUNTER_LIST="$COUNTER_LIST,$counter"
    fi
done

# Output file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="perf_stats_${TIMESTAMP}.txt"

echo "=========================================="
echo "Performance Counter Collection"
echo "=========================================="
echo "Counter file: $COUNTER_FILE"
echo "Counters: ${#COUNTERS[@]} counter(s) loaded"
echo "Python: $VENV_PYTHON"
echo "Command: $@"
echo "Output: $OUTPUT_FILE"
echo "=========================================="
echo ""

# Run perf stat with all NNPA counters
# Using sudo with full path to virtual environment Python
sudo perf stat \
    -e "$COUNTER_LIST" \
    -o "$OUTPUT_FILE" \
    "$@"
#    "$VENV_PYTHON" "$@"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Collection completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "Summary of results:"
    echo "=========================================="
    cat "$OUTPUT_FILE"
else
    echo "Collection failed with exit code: $EXIT_CODE"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Partial results may be in: $OUTPUT_FILE"
    fi
fi
echo "=========================================="

exit $EXIT_CODE

# Made with Bob
