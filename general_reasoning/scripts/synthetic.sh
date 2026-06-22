#!/bin/bash
set -e

# Load .env from repo root if it exists
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs)
fi

# ---------------- Configuration ----------------
MAIN_MODEL_NAME="${GR_SYN_MODEL_NAME:?MODEL_NAME must be set (env or .env)}"
BASE_URL="${GR_SYN_BASE_URL:?BASE_URL must be set (env or .env)}"
API_KEY="${GR_SYN_API_KEY:?API_KEY must be set (env or .env)}"

DATASET_PATH="${DATASET_PATH:?DATASET_PATH must be set to an input JSONL file}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
OUTPUT_BASE_NAME="${OUTPUT_BASE_NAME:-run_$(date +%Y%m%d_%H%M)}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-system_prompt_math_unified}"
TOOLSETS="${TOOLSETS:-TOOLS_UNIFIED_FINAL_1229}"

MAX_WORKERS="${MAX_WORKERS:-20}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

# Run from the synthetic/ directory so main.py and relative paths work correctly
cd "$SCRIPT_DIR/../synthetic"

TOTAL=$(python3 -c "
import json, sys
count = sum(1 for line in open('$DATASET_PATH') if line.strip())
print(count)
")
echo "Total examples in dataset: $TOTAL"

CHUNK_SIZE=$(( (TOTAL + NUM_PROCESSES - 1) / NUM_PROCESSES ))
echo "Each process chunk size: $CHUNK_SIZE"

# ---------------- Launch processes ----------------
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    START_IDX=$(( i * CHUNK_SIZE ))
    END_IDX=$(( START_IDX + CHUNK_SIZE ))
    if [ "$END_IDX" -gt "$TOTAL" ]; then
        END_IDX=$TOTAL
    fi

    OUTPUT_DIR="$OUTPUT_ROOT/proc_$i"
    mkdir -p "$OUTPUT_DIR"

    echo "Launching process $i: samples $START_IDX ~ $END_IDX"

    python3 -u main.py \
        --task_name GeneralReasoning \
        --model_name "$MAIN_MODEL_NAME" \
        --base_url "$BASE_URL" \
        --api_key "$API_KEY" \
        --dataset_path "$DATASET_PATH" \
        --toolsets "$TOOLSETS" \
        --system_prompt "$SYSTEM_PROMPT" \
        --output_dir "$OUTPUT_DIR" \
        --output_base_name "$OUTPUT_BASE_NAME" \
        --max_workers "$MAX_WORKERS" \
        --bidx "$START_IDX" \
        --eidx "$END_IDX" &
done

wait
echo "All processes finished."
