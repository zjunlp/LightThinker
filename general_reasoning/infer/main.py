import argparse
import logging
import os
import json
import json5
import re
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from tqdm import tqdm

from .benchmark import run_benchmark

def setup_logger(log_dir: str, run_args: argparse.Namespace) -> str:
    """、。"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(str(run_args.reasoning_model).rstrip('/'))
    dataset_name = run_args.dataset_name
    log_file = f"{log_dir}/log_{model_name}_{dataset_name}_{timestamp}.log"
    
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")
    logging.info(f"Run arguments: {vars(run_args)}")
    return log_file

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Self-Contained GPQA-Diamond Benchmark")
    
    parser.add_argument("--reasoning_model", type=str, required=True, help="Path to the vLLM-compatible model directory.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer directory. If None, defaults to reasoning_model path.")
    parser.add_argument("--tensor_parallel_size", "-tp", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=10240)
    parser.add_argument("--turn_max_tokens", type=int, default=1024)
    parser.add_argument("--max_tokens_setting", type=str, default="dynamic", choices=["budget", "throughput"], help="Strategy for setting max tokens per turn.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_num", type=int, default=2)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--system_prompt", action="store_true", help="Whether to include the system prompt in the model input.")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty for the model generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty for the model generation.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    
    parser.add_argument("--dataset_name", type=str, default="gpqa", help="Name of the benchmark dataset to use.")
    parser.add_argument("--subset_name", type=str, default=None, help="Subset name for datasets that support subsets (e.g., MMLU).")
    
    # mode options
    parser.add_argument("--mode", type=str, choices=["vllm", "openai", "hf"], default="hf", help="Inference mode to use.")
    parser.add_argument("--inference_type", type=str, choices=["interactive", "vanilla_cot", "token_skip"], default="interactive", help="Type of inference to perform.")
    parser.add_argument("--disable_expand", action="store_true", help="Disable the expand action during inference.")
    parser.add_argument("--disable_fold", action="store_true", help="Disable the fold action during inference.")

    # infer specific args
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    
    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.reasoning_model

    _log_dir = os.path.join(args.output_dir, "logs")
    setup_logger(log_dir=_log_dir, run_args=args)

    try:
        run_benchmark(**vars(args))
    except Exception as e:
        logging.error("A critical error occurred during the benchmark run.", exc_info=True)

if __name__ == "__main__":
    main()
