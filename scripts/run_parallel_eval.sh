#!/bin/bash
# Run multiple solo evaluations in parallel
# Usage: ./scripts/run_parallel_eval.sh models.txt [config.yaml]
# Each line in the input file should be a model key from models.yaml

set -e

# Check for input file argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <models_file> [config_file]"
    echo "  models_file: Each line should be a model key from models.yaml"
    echo "  config_file: Optional, defaults to config.yaml"
    echo ""
    echo "Example:"
    echo "  $0 eval_models.txt"
    echo "  $0 eval_models.txt test.yaml"
    exit 1
fi

MODELS_FILE="$1"
CONFIG_FILE="${2:-config.yaml}"

if [ ! -f "$MODELS_FILE" ]; then
    echo "Error: Models file not found: $MODELS_FILE"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Read models from file (skip empty lines and comments)
MODELS=()
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    line=$(echo "$line" | sed 's/#.*//' | xargs)
    [ -z "$line" ] && continue
    MODELS+=("$line")
done < "$MODELS_FILE"

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models found in $MODELS_FILE"
    exit 1
fi

echo "Running parallel evaluation for ${#MODELS[@]} models..."
echo "Config: $CONFIG_FILE"
echo "Models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Show time estimate based on historical results
if command -v uv &>/dev/null; then
    uv run python scripts/estimate_parallel_time.py "$MODELS_FILE" "$CONFIG_FILE" 2>/dev/null || true
elif command -v python3 &>/dev/null; then
    python3 scripts/estimate_parallel_time.py "$MODELS_FILE" "$CONFIG_FILE" 2>/dev/null || true
fi

for model in "${MODELS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Starting: $model"
    uv run python scripts/evaluate_single.py --config "$CONFIG_FILE" --model "$model" &
    sleep 2
done

wait
echo ""
echo "Done! Results in: results/"
