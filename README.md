<div align="center" style="width: 100%;">
  <img src="assets/gif.gif" alt="Case GIF" style="width: 100%; max-width: 100%; height: auto;">
</div>



<div align="center">
<h1 align="center"> 👉 LightThinker 👈 </h1>
<b>LightThinker: Thinking Step-by-Step Compression</b>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/LightThinker) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/LightThinker?color=green) 

<p align="center">
  <a href="https://arxiv.org/abs/2502.15589">📄arXiv</a> •
  <a href="https://x.com/zxlzr/status/1894729164609208338">𝕏 Blog</a> •
  <a href="https://huggingface.co/collections/zjunlp/lightthinker-67f9faaaa518f2e00b17386b">🤗 Huggingface</a>
</p>

</div>

## Table of Contents

- 👀[Overview](#overview)
- 🔧[Installation](#installation)
- 🏃[Quick Start](#quick-start)
- 🎁[Acknowledgement](#acknowledgement)
- 🚩[Citation](#citation)

## 👀Overview

LLMs have shown remarkable performance in complex reasoning tasks, but their efficiency is hindered by the substantial memory and computational costs associated with generating lengthy tokens. In this paper, we propose LightThinker, a novel method that enables LLMs to dynamically compress intermediate thoughts during reasoning. Inspired by human cognitive processes, LightThinker compresses verbose thought steps into compact representations and discards the original reasoning chains, thereby significantly reducing the number of tokens stored in the context window. This is achieved by training the model on when and how to perform compression through data construction, mapping hidden states to condensed gist tokens, and creating specialized attention masks.


## 🔧Installation

```bash
git clone https://github.com/zjunlp/LightThinker
cd LightThinker
conda create -n lightthinker python=3.9 -y
conda activate lightthinker
pip install -r requirements.txt
cd data && unzip data.zip && cd ..
```


## 🏃Quick Start

> First, we train the model to learn how to compress (step 1). Then, we perform inference on the test set to obtain output results (step 2). Finally, we evaluate the output results (step 3).

### Step 1. Training

To execute the training, run the following command:

```bash
bash train.sh
```

Currently, the script's parameters are set to run on a machine with 4 A800 GPUs. If you encounter OOM (Out Of Memory) issues, please reduce the `micro_batch_size` and `max_length`. For other parameters in the script, please refer to the [documentation](./ARGS.md).

### Step 2. Inference

<details> 
<summary><b>Inference with a downloaded model</b></summary>

If you are downloading a trained model from Huggingface, please set the `model_path` parameter in `inference.sh` to the absolute path of the model. The values of other parameters `ckpt` and `model_tag` will be ignored.
</details>

To execute the inference, run the following command:

```bash
bash inference.sh
```

Here, you need to modify the script file's `model_tag`, `model_short_tag`, `ckpt`, `output_tag`, and `split_size`. For details regarding the script's parameters, please refer to the [documentation](./ARGS.md).


### Step 3. Evaluation

> [!NOTE]
> If this is your **first time** conducting an evaluation, please execute the following code first:
> ```bash
> python evaluation/init.py
> ```

To execute the evaluation, run the following command:

```bash
method=""
tokenizer_path=""
comp_config=""
model_type=""
dataset=""
bos_token=""
eos_token=""
cache_size=1024
file1=""
file2=""
file3=""
file4=""
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

Please note that if you set `split_size>1` in the second step, the number of file i here should match the value of `split_size`. It should be noted that manual evaluation was conducted during the assessment. Use the `--interaction` flag to enable manual evaluation. The `cache_size` parameter is used for `H2O` and `SepLLM`, but not for `LightThinker` or `AnLLM`.

<details> 
<summary><b>Evaluation Script Example</b></summary>

```bash
# The optional values for the method argument are 'anchor-token', 'normal', 'kvcache', and 'anchor-thought'.
method="anchor-thought"
tokenizer_path="Qwen/Qwen2.5-7B-Instruct"
comp_config="configs/LightThinker/qwen/v1.json"
model_type="qwen"
dataset="gpqa"
bos_token="<|im_start|>"
eos_token="<|im_end|>"
cache_size=1024
folder=""
ckpt=1045
file1="inference_results/${folder}/${dataset}/${ckpt}/1-4qwen_7b.jsonl"
file2="inference_results/${folder}/${dataset}/${ckpt}/2-4qwen_7b.jsonl"
file3="inference_results/${folder}/${dataset}/${ckpt}/3-4qwen_7b.jsonl"
file4="inference_results/${folder}/${dataset}/${ckpt}/4-4qwen_7b.jsonl"
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
</details>

<details> 
<summary><b>Manual Evaluation Instructions</b></summary>

When string matching fails, the output will be displayed in the format "Model Answer" <=> "Standard Answer". At this point, you can input "y" or "n" to evaluate this case. If you believe the model's answer extraction is incorrect, you can input "e" to print the model's complete output, and then input "y" or "n" to evaluate this case.
</details>



## 🎁Acknowledgement

Our training dataset is derived from [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k). We utilized the baseline code for H2O from the [Meta-llama](https://github.com/meta-llama/llama-cookbook)'s repository, the baseline code for SepLLM from the [HKUDS](https://github.com/HKUDS/SepLLM)'s repository. We extend our gratitude to the contributors for their outstanding work!


## 🚩Citation

If this work is helpful, please kindly cite as:

```bibtex
@article{DBLP:journals/corr/abs-2502-15589,
  author       = {Jintian Zhang and
                  Yuqi Zhu and
                  Mengshu Sun and
                  Yujie Luo and
                  Shuofei Qiao and
                  Lun Du and
                  Da Zheng and
                  Huajun Chen and
                  Ningyu Zhang},
  title        = {LightThinker: Thinking Step-by-Step Compression},
  journal      = {CoRR},
  volume       = {abs/2502.15589},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2502.15589},
  doi          = {10.48550/ARXIV.2502.15589},
  eprinttype    = {arXiv},
  eprint       = {2502.15589},
  timestamp    = {Thu, 20 Mar 2025 13:28:42 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2502-15589.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```





