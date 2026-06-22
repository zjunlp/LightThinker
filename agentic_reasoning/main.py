# main.py

import argparse
from core.tasks.deepresearch import DeepResearchDataset
import logging
from logging import StreamHandler, FileHandler
import sys
import os
from typing import Union
import datetime

def setup_logger(log_file: str = None, log_dir: str = ".", model_name: str = "deepseek",  task_name="DeepResearch", run_args: Union[dict, object, None] = None):
    """
    :
    - (WARNING)
    - (INFO)
    - 
    - : 
    
    Args:
        log_file:  ()
        log_dir:  ()
        run_args: argparse.Namespace  dict，
    Returns:
        log_path: 
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if getattr(logger, "_is_logger_initialized", False):
        return getattr(logger, "_log_file_path", None)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console_handler = StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    
    if log_file is None:
        log_file = f"{task_name}_{model_name}.log"
    if ".log" not in log_file:
        log_file += timestamp + f".log"
    

    log_path = os.path.join(log_dir, log_file)

    file_handler = FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info(f"[Logger] Logging to file: {log_path}")

    if run_args is not None:
        try:
            if not isinstance(run_args, dict):
                run_args = vars(run_args)
        except Exception:
            run_args = run_args.__dict__
        logging.info("[Run Args] ：")
        for k, v in run_args.items():
            logging.info(f"    {k}: {v}")

    logger._is_logger_initialized = True
    logger._log_file_path = log_path

    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPQA-Diamond benchmark")
    parser.add_argument("--task_name", type=str,
                        default="DeepResearch",
                        choices=["DeepResearch", "BC_en", "BC_zh", "xbench"],
                        help="Name of the benchmark task to run")
    parser.add_argument("--dataset_path", type=str,
                        default="./datasets/test.jsonl",
                        help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str,
                        default="results/",
                        help="Directory to save output results")
    parser.add_argument("--output_base_name", type=str,
                        default="test",
                        help="Base name for output files")
    parser.add_argument("--toolsets", type=str,
                        default="TOOLS1",
                        help="Which toolset to use for the benchmark")
    parser.add_argument("--bidx", type=int, default=0,
                        help="begin index of the dataset to run")
    parser.add_argument("--eidx", type=int, default=-1,
                        help="end index of the dataset to run")
    parser.add_argument("--model_name", type=str,
                        default="Qwen3-8B",
                        help="model name or path served by vLLM")
    parser.add_argument("--base_url", type=str,
                        help="base url of the api server")
    parser.add_argument("--api_key", type=str, default="",
                        help="api key for the llm service")
    parser.add_argument("--system_prompt", type=str,
                    default="default",
                    help="Which system prompt to use for the datasets")
    parser.add_argument("--main_system_prompt", type=str,    
                        default=None,
                        help="Which system prompt to use for the main model")
    parser.add_argument("--logger_system_prompt", type=str,    
                        default=None,
                        help="Which system prompt to use for the logger model")
    parser.add_argument("--search_logger_system_prompt", type=str,    
                        default=None,
                        help="Which system prompt to use for the search logger model")
    parser.add_argument("--visit_logger_system_prompt", type=str,
                        default=None,
                        help="Which system prompt to use for the visit logger model")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Maximum number of parallel workers for generation and evaluation")
    
    
    parser.add_argument(
        "--react_mode",
        type=str,
        default="prompt",
        choices=["prompt", "fn_call"],
        help=(
            "Which ReAct / tool-calling mode to use in DeepResearchDataset:\n"
            "'prompt' - model outputs <tool_call>...</tool_call> text, then we parse and exec tools;\n"
            "'fn_call' - use OpenAI-style tools/function calling API."
        )
    )

    # =================================
    # =================================
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="sampling temperature passed to GPTChatter")
    parser.add_argument("--max_tokens", type=int, default= 32768,
                        help="max tokens to generate per request 30000")
    parser.add_argument("--n_repeat", type=int, default=1,
                        help="how many times to query the model for each question")
    parser.add_argument("--debug", action="store_true",
                        help="load only a handful of samples for quick debugging")
    parser.add_argument("--debug_num", type=int, default=2,
                        help="How many questions to run when --debug is set")
    parser.add_argument("--enable_thinking", type=bool, default=True,
                        help="Enable thinking in the model responses")    


    args = parser.parse_args()

    prefix = ""
    output_base_dir_name=""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    prefix = os.path.join(args.output_dir, args.model_name)
    if args.output_base_name != "test":
        output_base_name = args.output_base_name 
    else:
        output_base_name = timestamp + "_" + args.toolsets + "_" + args.system_prompt
    output_base_dir_name = os.path.join(prefix, output_base_name)

    log_dir = os.path.join(prefix, "logs")
    log_file = output_base_name.replace(f"_{timestamp}", "") 
    log_path = setup_logger(log_dir=log_dir, log_file=log_file, model_name= args.model_name, task_name=args.task_name, run_args=args)  
    logging.info(f"[Main] : {os.path.abspath(log_path)}")
    print(f"[Main] : {os.path.abspath(log_path)}")

    all_args = vars(args)
    task_args = all_args.copy()
    task_args['output_base_name'] = output_base_dir_name 
    
    if args.task_name in ["DeepResearch", "BC_en", "BC_zh", "xbench"]:
        run_task = DeepResearchDataset(**task_args)
    else:
        raise ValueError(f": {args.task_name}")

    run_task.eval()
    
    logging.info(f"[Main] ，: {os.path.abspath(log_path)}")
    print(f"[Main] ，: {os.path.abspath(log_path)}")
    # ==========================================================

    

if __name__ == "__main__":
    main()
