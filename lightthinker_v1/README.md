# LightThinker

LightThinker compresses intermediate thoughts into compact gist-token representations. This directory contains the LightThinker implementation, the AnLLM baseline, compression configs, evaluation utilities, and train / inference scripts.

---

## Models and Data

| Type | Name | Location |
|------|------|----------|
| Model | LightThinker-Llama | [zjunlp/LightThinker-Llama](https://huggingface.co/zjunlp/LightThinker-Llama) |
| Model | LightThinker-Qwen | [zjunlp/LightThinker-Qwen](https://huggingface.co/zjunlp/LightThinker-Qwen) |
| Data | Data archive | [`data/data.zip`](./data/data.zip) |

After entering this directory, the data archive path is `data/data.zip`.

---

## Environment Setup

```bash
conda create -n lightthinker python=3.9 -y
conda activate lightthinker
pip install -r requirements.txt
```

If needed, unzip the data archive before training or evaluation:

```bash
cd data
unzip data.zip
cd ..
```

---

## Training

Run from `lightthinker_v1/`:

```bash
bash train.sh
```

The default script configuration targets a machine with 4 A800 GPUs. If you encounter OOM issues, reduce `micro_batch_size` and `max_length`. See [`ARGS.md`](./ARGS.md) for all script arguments.

---

## Inference

Run from `lightthinker_v1/`:

```bash
bash inference.sh
```

To use a downloaded model, set `model_path` in `inference.sh`; then `ckpt` and `model_tag` are ignored.

For the AnLLM baseline, use:

```bash
bash inference_anllm.sh
```

---

## Evaluation

Run the initialization step once before your first evaluation:

```bash
python evaluation/init.py
```

Example evaluation command:

```bash
method="anchor-thought"
tokenizer_path="Qwen/Qwen2.5-7B-Instruct"
comp_config="configs/LightThinker/qwen/v1.json"
model_type="qwen"
dataset="gpqa"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size=1024
file1="inference_results/${dataset}/1-4qwen_7b.jsonl"
file2="inference_results/${dataset}/2-4qwen_7b.jsonl"
file3="inference_results/${dataset}/3-4qwen_7b.jsonl"
file4="inference_results/${dataset}/4-4qwen_7b.jsonl"
python evaluation/eval_file.py \
  --method $method \
  --tokenizer_path $tokenizer_path \
  --comp_config $comp_config \
  --model_type $model_type \
  --dataset $dataset \
  --files $file1 $file2 $file3 $file4 \
  --cache_size $cache_size \
  --bos_token $bos_token \
  --eos_token $eos_token \
  --interaction
```

If `split_size > 1` during inference, pass the same number of result files to evaluation.
