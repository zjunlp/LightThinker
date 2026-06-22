# Agentic Reasoning

Synthetic data generation and inference pipeline for deep-research / multi-hop QA tasks: BrowseComp-en, BrowseComp-zh, HotpotQA-MuSiQue, and xBench.

The agent uses a **Hierarchical Memory System** — `search`, `visit`, `expand`, `fold`, and `commit` tools — to conduct multi-step web research while keeping the context window manageable.

---

## Models and Data

| Type | Name | Hugging Face |
|------|------|--------------|
| Model | LightThinker-Plus-Qwen3-30B-A3B | [zjunlp/LightThinker-Plus-Qwen3-30B-A3B](https://huggingface.co/zjunlp/LightThinker-Plus-Qwen3-30B-A3B) |
| SFT Data | Agentic Reasoning | [zjunlp/LightThinker-Plus-AgenticReasoning-SFT](https://huggingface.co/datasets/zjunlp/LightThinker-Plus-AgenticReasoning-SFT) |

---

## Environment Setup

Install dependencies:

```bash
bash setup.sh
```

---

## API Keys

Two external services are required for web search and page reading:

| Service | Purpose | Sign-up |
|---------|---------|---------|
| [Serper](https://serper.dev/) | Web search & Google Scholar | `SERPER_KEY_ID` |
| [Jina AI](https://jina.ai/) | Webpage content parsing | `JINA_API_KEYS` |

---

## Configuration

The pipeline is controlled entirely through environment variables loaded from a `.env` file.
Two commit-safe template files are provided — copy the appropriate one before running:

```bash
# For synthetic data generation:
cp .env.synthetic.example .env

# For inference on a trained model:
cp .env.infer.example .env
```

Then open `.env` and fill in your credentials.

### Required variables

| Variable | Description |
|----------|-------------|
| `SERPER_KEY_ID` | Serper API key |
| `JINA_API_KEYS` | Jina AI API key |
| `REASONING_SK` | API key for the main reasoning model |
| `REASONING_ENDPOINT` | OpenAI-compatible base URL for the main model |
| `MODEL_NAME` | Model name as served by the endpoint |

### Optional variables (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ROUNDS` | `100` | Maximum agent turns per sample |
| `MAX_REASON_TOKENS` | `110592` | Token budget before forcing a final answer |
| `LOGGER_MODEL_NAME` | `""` | Logger / summarizer model name |
| `LOGGER_MODEL_URL` | `""` | Logger model API endpoint |
| `LOGGER_MODEL_API_KEY` | `""` | Logger model API key |
| `EVALUATE_MODEL_NAME` | `""` | Judge model for evaluation |
| `EVALUATE_API_BASE` | `""` | Judge model API endpoint |
| `EVALUATE_API_KEY` | `""` | Judge model API key |
| `TOKENIZER_PATH` | `./core/tasks/tokenizer/Qwen3-30B-A3B-Thinking-2507` | HuggingFace tokenizer for token counting (falls back to tiktoken if unavailable) |

---

## Running Synthetic Data Generation

Edit the user-config section at the top of `scripts/synthetic.sh`, then:

```bash
bash scripts/synthetic.sh
```

To override the model or run multiple iterations:

```bash
bash scripts/synthetic.sh \
    --ports 8210 \
    --model_name your-model-name \
    --max_workers 10 \
    --iterations 3
```

**Key script variables:**

| Variable | Description |
|----------|-------------|
| `EXPERIMENT_NAME` | Output directory name under `results-synthetic/` |
| `TASK_NAME` | Dataset task: `DeepResearch`, `BC_en`, `BC_zh`, `xbench` |
| `DATASET_PATH` | Path to the input JSONL dataset |
| `MAIN_SYS_PROMPT` | Key into `PROMPTS_MAP` for the main agent |
| `SEARCH_LOGGER_SYS_PROMPT` | Key for the search-logger agent |
| `VISIT_LOGGER_SYS_PROMPT` | Key for the visit-logger agent |
| `REACT_MODE` | `fn_call` (recommended) or `prompt` |

Output is written to `results-synthetic/<EXPERIMENT_NAME>/iter<N>_raw_outputs.jsonl`.

---

## Running Inference

Edit the model settings in `.env`, then:

```bash
bash scripts/inference.sh
```

The inference script uses a single `system_prompt` (no dual-model logger):

```bash
bash scripts/inference.sh \
    --ports 8210 \
    --model_name your-model-name \
    --max_workers 10
```

Output is written to `results-inference/<EXPERIMENT_NAME>/iter<N>_raw_outputs.jsonl`.

---

## Data Post-Processing

Convert raw interaction trajectories into SFT training data:

```bash
python process/data_filter.py \
    --input results-synthetic/my-run/iter1_raw_outputs.jsonl \
    --output curated/train_data.json
```

Multiple input files can be provided:

```bash
python process/data_filter.py \
    --input results-synthetic/run1/iter1_raw_outputs.jsonl \
              results-synthetic/run2/iter1_raw_outputs.jsonl \
    --output curated/train_data.json
```

The script applies the following quality filters:

- **Short simple trajectories** (< 3 rounds with no memory tools) are dropped
- **Memory logic validation**: trajectories using `expand`/`fold`/`commit` must satisfy `expand → fold` ordering with no duplicates
- **Length filter**: trajectories exceeding 60 rounds are dropped

**Output report columns:**

| Column | Description |
|--------|-------------|
| `Memory Kept` | Valid trajectories using expand/fold/commit |
| `Simple Kept` | Valid simple trajectories (no memory tools) |
| `Drop (No EF Mix)` | Missing required expand+fold+commit combination |
| `Drop (Fold Error)` | fold called without a preceding expand |
| `Drop (Jitter)` | Consecutive duplicate memory operations |
| `Drop (Too Long)` | Exceeded 60-round threshold |

---

## Direct Usage (Python)

```bash
# fn_call mode (recommended for synthetic data generation)
python main.py \
    --task_name DeepResearch \
    --model_name your-model \
    --base_url http://localhost:8000/v1 \
    --dataset_path datasets/browsecomp_en.jsonl \
    --toolsets DEEPRESEARCH \
    --react_mode fn_call \
    --main_system_prompt synthetic_main_agent \
    --search_logger_system_prompt search_logger \
    --visit_logger_system_prompt visit_logger \
    --output_dir results/ \
    --output_base_name iter1 \
    --max_workers 10

# prompt mode (for inference with a single system prompt)
python main.py \
    --task_name DeepResearch \
    --model_name your-model \
    --base_url http://localhost:8000/v1 \
    --dataset_path datasets/browsecomp_en.jsonl \
    --toolsets DEEPRESEARCH \
    --react_mode prompt \
    --system_prompt infer_agent \
    --output_dir results/ \
    --output_base_name iter1
```

**Supported `--task_name` values:** `DeepResearch`, `BC_en`, `BC_zh`, `xbench`

**Supported `--toolsets` values:** `DEEPRESEARCH`, `DEEPRESEARCH_BASE`

---

## Supported Datasets

Place dataset files under `datasets/`. Expected JSONL format — one record per line with at minimum `id`, `question`, and `answer` fields.
