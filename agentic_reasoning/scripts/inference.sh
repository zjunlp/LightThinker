#!/bin/bash
set -euo pipefail

# ==========================================
# ==========================================
EXPERIMENT_NAME="SFT_Lthinker_Qwen30bThinking"
TASK_NAME="DeepResearch"
DATASET_PATH="./datasets/test.jsonl"
TOOLSETS="DEEPRESEARCH"
SYS_PROMPT="infer_agent"
OUTPUT_DIR="results-inference/"
REACT_MODE="prompt"
TEMP="0.6"
DPTOOL_ONLY_FLAG=""
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(dirname "$SCRIPT_DIR")"
WORKING_ROOT="$APP_ROOT"
ENV_FILE="$APP_ROOT/.env"

PORTS_CSV=""
CLI_MODEL_NAME=""
MAX_WORKERS=1
NUM_ITERATIONS=1

CHILD_PIDS=()

usage() {
    cat <<EOF
: $0 []

:
  --ports P1,P2,...   （ iter ，）
  --model_name NAME    .env  MODEL_NAME
  --max_workers W      worker （ 10）
  --iterations N       N （ 1）
  -h, --help          
EOF
}

cleanup() {
    echo -e "\n\n... ..."
    if [ ${#CHILD_PIDS[@]} -gt 0 ]; then
        for pid in "${CHILD_PIDS[@]}"; do
            if ps -p "$pid" > /dev/null 2>&1; then
                echo ": $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        sleep 2
        for pid in "${CHILD_PIDS[@]}"; do
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
    wait 2>/dev/null || true
    echo "，。"
    exit 1
}
trap cleanup SIGINT SIGTERM

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ports|--model_name|--max_workers|--iterations)
            if [[ -z "${2:-}" || "$2" =~ ^- ]]; then
                echo ":  '$1' 。" >&2; usage; exit 1
            fi
            case "$1" in
                --ports)       PORTS_CSV="$2" ;;
                --model_name)  CLI_MODEL_NAME="$2" ;;
                --max_workers) MAX_WORKERS="$2" ;;
                --iterations)  NUM_ITERATIONS="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo ": $1" >&2; usage; exit 1 ;;
    esac
done

if [ ! -f "$ENV_FILE" ]; then
    echo ": .env : $ENV_FILE"; exit 1
fi
echo " $ENV_FILE ..."
set -a
source "$ENV_FILE"
set +a

if [ -n "$CLI_MODEL_NAME" ]; then MODEL_NAME="$CLI_MODEL_NAME"; fi

if [ -z "${MODEL_NAME:-}" ] || [ -z "${REASONING_SK:-}" ]; then
    echo ":  (MODEL_NAME, REASONING_SK) 。"; exit 1
fi

if ! [[ "$MAX_WORKERS" =~ ^[0-9]+$ ]] || (( MAX_WORKERS <= 0 )); then
    echo ": max_workers ，: $MAX_WORKERS" >&2; exit 1
fi
if ! [[ "$NUM_ITERATIONS" =~ ^[0-9]+$ ]] || (( NUM_ITERATIONS <= 0 )); then
    echo ": iterations ，: $NUM_ITERATIONS" >&2; exit 1
fi

PORTS=()
if [ -n "$PORTS_CSV" ]; then
    IFS=',' read -r -a PORTS <<< "$PORTS_CSV"
    for p in "${PORTS[@]}"; do
        if ! [[ "$p" =~ ^[0-9]+$ ]]; then
            echo ": --ports : $p" >&2; exit 1
        fi
    done
else
    if [ -z "${REASONING_ENDPOINT:-}" ]; then
        echo ":  --ports  .env  REASONING_ENDPOINT" >&2; exit 1
    fi
fi

get_endpoint() {
    local iter_num=$1
    if [ ${#PORTS[@]} -gt 0 ]; then
        local port_index=$(( (iter_num - 1) % ${#PORTS[@]} ))
        echo "http://127.0.0.1:${PORTS[$port_index]}/v1"
    else
        echo "${REASONING_ENDPOINT}"
    fi
}

run_task() {
    local iter_num=$1
    local endpoint
    endpoint=$(get_endpoint "$iter_num")

    local experiment_dir="${OUTPUT_DIR}/${EXPERIMENT_NAME}/"
    local output_base_name="iter${iter_num}"
    local log_dir="${experiment_dir}/logs"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/${output_base_name}.log"

    CMD_ARRAY=(
        "python" "-u" "main.py"
        "--task_name"        "$TASK_NAME"
        "--model_name"       "$MODEL_NAME"
        "--base_url"         "$endpoint"
        "--api_key"          "$REASONING_SK"
        "--dataset_path"     "$DATASET_PATH"
        "--toolsets"         "$TOOLSETS"
        "--temperature"      "$TEMP"
        "--system_prompt"    "$SYS_PROMPT"
        "--output_dir"       "$experiment_dir"
        "--output_base_name" "$output_base_name"
        "--react_mode"       "$REACT_MODE"
        "--max_workers"      "$MAX_WORKERS"
    )
    if [ -n "$DPTOOL_ONLY_FLAG" ]; then CMD_ARRAY+=("$DPTOOL_ONLY_FLAG"); fi

    echo "----------------------------------------------------------"
    echo ": iter=${iter_num}/${NUM_ITERATIONS} endpoint=${endpoint}"
    echo "log=${log_file}"
    echo "----------------------------------------------------------"
    ( "${CMD_ARRAY[@]}" 2>&1 | tee "$log_file" )
}

echo "=========================================================="
echo ":"
echo "  EXPERIMENT_NAME = ${EXPERIMENT_NAME}"
echo "  TASK_NAME       = ${TASK_NAME}"
echo "  REACT_MODE      = ${REACT_MODE}"
echo "  iterations      = ${NUM_ITERATIONS}"
echo "  max_workers     = ${MAX_WORKERS}"
if [ ${#PORTS[@]} -gt 0 ]; then
    echo "  ports           = ${PORTS_CSV} (/)"
else
    echo "  endpoint        = ${REASONING_ENDPOINT:-}"
fi
echo "=========================================================="

cd "$WORKING_ROOT" || { echo ": $WORKING_ROOT"; exit 1; }
mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}/logs"

for (( iter=1; iter<=NUM_ITERATIONS; iter++ )); do
    run_task "$iter"
done

echo "=========================================================="
echo "。: ${OUTPUT_DIR}/${EXPERIMENT_NAME}/"
echo "=========================================================="
