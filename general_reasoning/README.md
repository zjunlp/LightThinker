# General Reasoning

End-to-end pipeline: synthetic data generation → SFT filtering → training → inference → evaluation.

---

## Models and Data

| Type | Name | Hugging Face |
|------|------|--------------|
| Model | LightThinker-Plus-Qwen | [zjunlp/LightThinker-Plus-Qwen](https://huggingface.co/zjunlp/LightThinker-Plus-Qwen) |
| Model | LightThinker-Plus-Llama | [zjunlp/LightThinker-Plus-Llama](https://huggingface.co/zjunlp/LightThinker-Plus-Llama) |
| SFT Data | General Reasoning | [zjunlp/LightThinker-Plus-GeneralReasoning-SFT](https://huggingface.co/datasets/zjunlp/LightThinker-Plus-GeneralReasoning-SFT) |

---

## Environment Setup

**Requirements:** Python 3.12, CUDA 12.x, PyTorch 2.7

**Step 1 — Create and activate the conda environment:**

```bash
conda env create -f environment.yml
conda activate lt_plus_general_reasoning
```

> The `environment.yml` uses Tsinghua mirrors. If you are outside China, replace the `channels` section with `- defaults` or `- conda-forge`.

**Step 2 — Install `flash_attn` via pre-built wheel:**

`flash_attn` is not included in `environment.yml` and must be installed separately. Download the wheel matching your CUDA / PyTorch / Python versions from:

https://github.com/Dao-AILab/flash-attention/releases

The wheel used in development:

```bash
# CUDA 12.x, PyTorch 2.7, Python 3.12, CXX11 ABI=TRUE
pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

**Step 3 - Set environment variables**

Copy `.env.example` to the repository root as `.env`, then modify it with your credentials and settings. The scripts auto-load `<repo_root>/.env`.
```bash
cp .env.example ../.env
set -a && source ../.env && set +a
```

---

## Raw Synthetic Data Generation

Configure credentials in `<repo_root>/.env` (auto-loaded by the script), set an input JSONL dataset, then run:

```bash
DATASET_PATH=/path/to/input.jsonl bash scripts/synthetic.sh
```

The script shards the dataset across `NUM_PROCESSES` parallel workers and runs `synthetic/main.py` on each shard.

**Required `.env` / env vars:**

| Env var | Description |
|---------|-------------|
| `GR_SYN_MODEL_NAME` | Model name served by the API endpoint |
| `GR_SYN_BASE_URL` | OpenAI-compatible API base URL |
| `GR_SYN_API_KEY` | API key |

**Required runtime env vars:**

| Env var | Description |
|---------|-------------|
| `DATASET_PATH` | Input JSONL dataset |

**Optional env vars (with defaults):**

| Env var | Default | Description |
|---------|---------|-------------|
| `OUTPUT_ROOT` | `results/` | Root directory for per-process output |
| `OUTPUT_BASE_NAME` | `run_<timestamp>` | Base name for output files |
| `SYSTEM_PROMPT` | `system_prompt_math_unified` | Key into `PROMPTS_MAP` |
| `TOOLSETS` | `TOOLS_UNIFIED_FINAL_1229` | Key into `TOOL_MAP` |
| `MAX_WORKERS` | `20` | Threads per process |
| `NUM_PROCESSES` | `4` | Number of parallel processes |

Output is written to `$OUTPUT_ROOT/proc_<i>/` for each shard.

---

## Filtering and Formatting for SFT

Convert raw synthetic trajectories into LLaMA-Factory-compatible training data:

```bash
python synthetic/data_curation/train_data_construct.py \
  --input  <path/to/raw_results.jsonl_or_dir> \
  --output synthetic/curated_data/ \
  --filtering \
  --mode both \
  --strict
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Input JSON/JSONL file or directory |
| `--output` | (required) | Output directory |
| `--mode` | `both` | Output format: `stepwise`, `full`, or `both` |
| `--filtering` | off | Enable quality filtering (length, correctness checks) |
| `--strict` | off | Drop entire trajectory on duplicate/near-duplicate reasoning steps (default: drop only the step) |
| `--add_think_tag` | off | Prepend `<tool_call>` tag to `commit`/`expand`/`fold` reasoning |
| `--original_file` | `None` | Original JSONL used for ID-level deduplication filtering |

---

## Training

Edit `MODEL_PATH`, `TRAIN_DATA_PATH`, and `SAVE_PATH` in `scripts/train.sh`, then run from `ms-swift/`:

```bash
# Make sure you are in the ms-swift/ directory
bash scripts/train.sh
```

The script runs `swift sft` with the following fixed settings:

| Setting | Value |
|---------|-------|
| GPUs | 8 (CUDA_VISIBLE_DEVICES 0–7) |
| Parallelism | ZeRO-3 DeepSpeed |
| Dtype | bfloat16 + Flash Attention |
| Train type | Full fine-tune |
| Max sequence length | 16384 |
| Epochs | 3 |
| Batch size | 4 per device |
| Gradient accumulation | 1 |
| Warmup | 5% |
| Save strategy | Per epoch (keep last 3) |

**Learning rate guidance:**
- `1e-5` — vanilla CoT (`--inference_type vanilla_cot`)
- `2e-5` — LThinker interactive / other modes

---

## Inference

Run via `scripts/infer.sh`. The script is controlled by environment variables:

| Env var | Required | Default | Description |
|---------|----------|---------|-------------|
| `MODEL` | yes | — | Path to the model directory |
| `MODEL_TAG` | no | `basename $MODEL` | Tag used in output directory naming |
| `RESULT_ROOT` | no | `<repo>/results/eval` | Root directory for all outputs |
| `INFER_MODE` | no | `vanilla` | `vanilla` or `interactive` |
| `TOKENS_MODE` | no | `throughput` | `throughput` or `budget` (interactive only) |
| `BATCH_SIZE` | no | `32` | Inference batch size |

**Inference modes:**

| Mode | Underlying args |
|------|----------------|
| `vanilla` | `--inference_type vanilla_cot --system_prompt` |
| `interactive` + `throughput` | `--inference_type interactive --turn_max_tokens 1024 --max_tokens_setting throughput` |
| `interactive` + `budget` | `--inference_type interactive --max_tokens_setting budget` |

**Usage:**

```bash
# Vanilla CoT (default)
MODEL=/path/to/model bash scripts/infer.sh <GPU_ID> gsm8k

# Multiple datasets (comma-separated)
MODEL=/path/to/model bash scripts/infer.sh 0 --datasets gsm8k,math,gpqa

# Interactive mode (throughput)
INFER_MODE=interactive MODEL=/path/to/model bash scripts/infer.sh 0 gsm8k

# Interactive mode (budget)
INFER_MODE=interactive TOKENS_MODE=budget MODEL=/path/to/model bash scripts/infer.sh 0 gsm8k

# Custom output directory
INFER_MODE=interactive RESULT_ROOT=/data/results MODEL=/path/to/model bash scripts/infer.sh 0 gpqa
```

Results are saved under `$RESULT_ROOT/$MODEL_TAG/$DATASET/<mode>_turn<N>_temp<T>_.../{run.log, ...}`.

---

## Evaluation

Aggregate inference results into a leaderboard and compute accuracy, peak tokens, dependency, and throughput:

```bash
python scripts/eval.py \
  --input_dir  path/to/results/ \
  --datasets   gsm8k,mmlu,gpqa,bbh \
  --dependency_mode ltthinker
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | (required) | Directory tree containing `*results*.json` files |
| `--datasets` | `gsm8k,mmlu,gpqa,bbh` | Comma-separated list of datasets to evaluate |
| `--dependency_mode` | `ltthinker` | Computation mode for dependency metric: `vanilla` or `ltthinker` |
| `--max_repeats` | `3` | Max result files to aggregate per model–dataset pair |
| `--allow_incomplete` | off | Skip strict sample-count check (useful for partial runs) |
| `--show_files` | off | Print per-file load details |
| `--out_leaderboard_json` | `<input_dir>/leaderboard.json` | Path for the aggregated leaderboard JSON |
| `--out_wrong_json` | `<input_dir>/wrong_cases.json` | Path for the wrong-case dump |

**Supported datasets and expected sample counts:**

| Dataset | Expected count |
|---------|---------------|
| `gsm8k` | 1319 |
| `mmlu` | 1027 |
| `gpqa` | 198 |
| `bbh` | 495 |

Files with a count mismatch are skipped unless `--allow_incomplete` is set.

**Output columns** (printed as a tab-separated table and saved to `leaderboard.json`):

`Model_Config · AVG_Acc · AVG_Time(h) · AVG_Sec/Samp · AVG_Peak · AVG_Dep(M) · AVG_Turn · AVG_GenTok`
and per-dataset variants of each metric.
