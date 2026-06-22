import argparse
import datetime
import logging
import os
import sys
from logging import FileHandler, StreamHandler
from typing import Union

from core.tasks.general_reasoning import GeneralReasoningDataset


def setup_logger(
    log_file: str = None,
    log_dir: str = ".",
    model_name: str = "deepseek",
    task_name: str = "GeneralReasoning",
    run_args: Union[dict, object, None] = None,
):
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
        log_file += timestamp + ".log"

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
        logging.info("[Run Args]")
        for k, v in run_args.items():
            logging.info(f"    {k}: {v}")

    logger._is_logger_initialized = True
    logger._log_file_path = log_path

    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data synthetic pipeline")
    parser.add_argument("--task_name", type=str, default="GeneralReasoning", choices=["GeneralReasoning"],
                        help="Name of the benchmark task to run")
    parser.add_argument("--dataset_path", type=str,
                        required=True,
                        help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save output results")
    parser.add_argument("--output_base_name", type=str, default="test",
                        help="Base name for output files")
    parser.add_argument("--toolsets", type=str, default="TOOLS_UNIFIED_FINAL_1229",
                        help="Which toolset to use for the benchmark")
    parser.add_argument("--bidx", type=int, default=0, help="begin index of the dataset to run")
    parser.add_argument("--eidx", type=int, default=-1, help="end index of the dataset to run")
    parser.add_argument("--model_name", type=str, default="Qwen3-8B",
                        help="model name or path served by vLLM")
    parser.add_argument("--base_url", type=str, help="base url of the api server")
    parser.add_argument("--api_key", type=str, default="", help="api key for the llm service")
    parser.add_argument("--system_prompt", type=str, default="system_prompt_math_unified",
                        help="Which system prompt to use for the datasets")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Maximum number of parallel workers for generation and evaluation")
    parser.add_argument("--base_notool_mode", action="store_true", default=False,
                        help="Whether to run in base mode without tool use")
    parser.add_argument("--accumulated_next_step_prefix", action="store_true", default=False,
                        help="Whether to accumulate next_step_prefix over multiple expand steps")
    parser.add_argument("--temperature", type=float, default=0.6, help="sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=30000,
                        help="max tokens to generate per request")
    parser.add_argument("--n_repeat", type=int, default=1,
                        help="how many times to query the model for each question")
    parser.add_argument("--debug", action="store_true",
                        help="load only a handful of samples for quick debugging")
    parser.add_argument("--debug_num", type=int, default=2,
                        help="How many questions to run when --debug is set")
    parser.add_argument("--enable_thinking", type=bool, default=True,
                        help="Enable thinking in the model responses")

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    prefix = args.output_dir
    if args.eidx != -1:
        output_base_name = f"{args.output_base_name}_{args.bidx}_{args.eidx}"
    else:
        output_base_name = args.output_base_name
    output_base_dir_name = prefix + "/" + output_base_name

    log_dir = prefix + "/" + "logs/"
    log_file = output_base_name.replace(f"_{timestamp}", "")
    log_path = setup_logger(
        log_dir=log_dir,
        log_file=log_file,
        model_name=args.model_name,
        task_name=args.task_name,
        run_args=args,
    )

    task_args = vars(args).copy()
    task_args["output_base_name"] = output_base_dir_name

    if args.task_name == "GeneralReasoning":
        run_task = GeneralReasoningDataset(**task_args)
    else:
        raise ValueError(f"Unknown task: {args.task_name}")

    run_task.eval()

    logging.info(f"[Main] Done. Log saved to: {os.path.abspath(log_path)}")
    print(f"[Main] Done. Log saved to: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
