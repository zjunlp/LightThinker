#!/usr/bin/env bash
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ================= Env-var config =================
# INFER_MODE: vanilla | interactive   (default: vanilla)
# TOKENS_MODE: throughput | budget    (default: throughput, interactive only)
# MODEL: path to model                (required)
# MODEL_TAG: tag for result dir       (default: basename of MODEL)
# RESULT_ROOT: output root dir        (default: <workdir>/results/eval)
INFER_MODE="${INFER_MODE:-vanilla}"
TOKENS_MODE="${TOKENS_MODE:-throughput}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# ================= Argument parsing =================
usage() {
  echo "Usage:"
  echo "  $0 <GPU_ID> --datasets gsm8k,math,algebra"
  echo "  $0 <GPU_ID> <DATASET_1> [DATASET_2 ... DATASET_N]"
  echo ""
  echo "Env vars:"
  echo "  MODEL=<path>                       (required)"
  echo "  MODEL_TAG=<tag>                    (default: basename of MODEL)"
  echo "  RESULT_ROOT=<dir>                  (default: <workdir>/results/eval)"
  echo "  INFER_MODE=vanilla|interactive     (default: vanilla)"
  echo "  TOKENS_MODE=throughput|budget      (default: throughput, interactive only)"
  echo "  BATCH_SIZE=<int>                   (default: 32)"
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

GPU="$1"
shift

DATASETS=()
if [[ "$1" == "--datasets" ]]; then
  if [ "$#" -lt 2 ]; then
    echo "Error: --datasets requires a value"
    exit 1
  fi
  IFS=',' read -r -a DATASETS <<< "$2"
  shift 2
else
  DATASETS=("$@")
fi

if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "Error: No datasets specified"
  exit 1
fi

if [[ "$INFER_MODE" != "vanilla" && "$INFER_MODE" != "interactive" ]]; then
  echo "Error: INFER_MODE must be 'vanilla' or 'interactive', got '${INFER_MODE}'"
  exit 1
fi

if [[ "$INFER_MODE" == "interactive" && "$TOKENS_MODE" != "throughput" && "$TOKENS_MODE" != "budget" ]]; then
  echo "Error: TOKENS_MODE must be 'throughput' or 'budget', got '${TOKENS_MODE}'"
  exit 1
fi

echo "Using GPU=${GPU}"
echo "Datasets: ${DATASETS[*]}"
echo "Inference mode: ${INFER_MODE}"
[[ "$INFER_MODE" == "interactive" ]] && echo "Tokens mode: ${TOKENS_MODE}"
# ====================================================

# Derive working directory from script location so paths are portable.
# python -m infer.main must be run from LThinker-general_reasoning/.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${MODEL:?MODEL must be set (e.g. export MODEL=/path/to/model)}"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL")}"
RESULT_ROOT="${RESULT_ROOT:-$WORKDIR/results/eval}"

TEMPS=(0.7)
PRESENCE_PENALTIES=(0.0)
REPETITION_PENALTIES=(1.05)
MAX_TURNS=(30)
REPEAT=1

# Build mode-specific args array
if [[ "$INFER_MODE" == "vanilla" ]]; then
  MODE_ARGS=(--inference_type vanilla_cot --system_prompt)
  MODE_TAG="vanilla"
elif [[ "$TOKENS_MODE" == "throughput" ]]; then
  MODE_ARGS=(--inference_type interactive --turn_max_tokens 1024 --max_tokens_setting throughput)
  MODE_TAG="interactive_throughput"
else
  MODE_ARGS=(--inference_type interactive --max_tokens_setting budget)
  MODE_TAG="interactive_budget"
fi

cd "$WORKDIR"
for DATASET in "${DATASETS[@]}"; do
  echo "==================== DATASET ${DATASET} ===================="
  for REPEAT_IDX in $(seq 1 "$REPEAT"); do
    echo "==================== REPEAT ${REPEAT_IDX} ===================="
    for MT in "${MAX_TURNS[@]}"; do
      for T in "${TEMPS[@]}"; do
        for P in "${PRESENCE_PENALTIES[@]}"; do
          for R in "${REPETITION_PENALTIES[@]}"; do
            RUN_TAG="${MODE_TAG}_turn${MT}_temp${T}_pres${P}_rep${R}_repeat${REPEAT_IDX}"
            OUTDIR="${RESULT_ROOT}/${MODEL_TAG}/${DATASET}/${RUN_TAG}"
            mkdir -p "$OUTDIR"

            echo "[${DATASET}] GPU=${GPU}  mode=${MODE_TAG}  max_turn=${MT}  T=${T} P=${P} R=${R}"

            CUDA_VISIBLE_DEVICES="$GPU" python -m infer.main \
              --reasoning_model "$MODEL" \
              --tensor_parallel_size 1 \
              --temperature "$T" \
              --presence_penalty "$P" \
              --max_tokens 10240 \
              --top_p 0.95 \
              --repetition_penalty "$R" \
              --max_turn "$MT" \
              --dataset_name "$DATASET" \
              --output_dir "$OUTDIR" \
              --gpu_memory_utilization 0.7 \
              "${MODE_ARGS[@]}" \
              --mode vllm \
              --batch_size "$BATCH_SIZE" | tee "${OUTDIR}/run.log"
          done
        done
      done
    done
  done
done
