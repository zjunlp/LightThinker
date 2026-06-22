import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..llm_client import LLMClient
from ..memory import HybridMemory
from ..prompt_maps import PROMPTS_MAP
from ..tool_maps import TOOL_EXECUTORS, TOOL_MAP


class BaseDataset(ABC):
    def __init__(self, **kwargs):
        self.task_name: str = kwargs.get("task_name")
        self.model_name: str = kwargs.get("model_name")
        self.dataset_path: str = kwargs.get("dataset_path")
        self.base_url: str = kwargs.get("base_url")
        self.api_key: str = kwargs.get("api_key")
        self.temperature: float = kwargs.get("temperature", 0.6)
        self.top_p: float = kwargs.get("top_p", 0.95)
        self.max_tokens: int = kwargs.get("max_tokens", 32768)
        self.bidx: int = kwargs.get("bidx", 0)
        self.eidx: int = kwargs.get("eidx", -1)
        self.debug: bool = kwargs.get("debug", False)
        self.debug_num: int = kwargs.get("debug_num", 2)
        self.output_base_name: str = kwargs.get("output_base_name")
        self.toolsets_key: str = kwargs.get("toolsets", "none")
        self.eval_model_name: str = kwargs.get("eval_model", "local")
        self.enable_thinking: bool = kwargs.get("enable_thinking", False)
        self.base_notool_mode: bool = kwargs.get("base_notool_mode", False)
        self.accumulated_next_step_prefix: bool = kwargs.get("accumulated_next_step_prefix", False)
        self.max_workers: int = kwargs.get("max_workers", 10)

        self.system_prompt_key: str = kwargs.get("system_prompt", "default")

        self.raw_output_file = f"{self.output_base_name}_raw_outputs.jsonl"
        self.final_report_file = f"{self.output_base_name}_final_report.json"
        os.makedirs(os.path.dirname(self.raw_output_file), exist_ok=True)
        logging.info(f"[{self.__class__.__name__}] Raw outputs will be saved to: {self.raw_output_file}")
        logging.info(f"[{self.__class__.__name__}] Final report will be saved to: {self.final_report_file}")

        self._thread_local = threading.local()
        self.chatter = LLMClient(
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )
        self.tools = TOOL_MAP.get(self.toolsets_key, [])
        assert self.tools is not None, f"Unknown toolsets key: {self.toolsets_key}"
        self.tool_executors = TOOL_EXECUTORS

        logging.info(f"[{self.__class__.__name__}] Initializing evaluation with model type: {self.eval_model_name}")

        self.eval_chatter = LLMClient(
            model_name=os.environ.get("EVALUATE_MODEL_NAME", ""),
            base_url=os.environ.get("EVALUATE_API_BASE", ""),
            api_key=os.environ.get("EVALUATE_API_KEY", ""),
            temperature=0.0,
            max_tokens=4096,
        )

        self.system_prompt = PROMPTS_MAP.get(self.system_prompt_key, "")
        if not self.system_prompt:
            logging.error(
                f"[{self.__class__.__name__}] Task '{self.task_name}' requires a non-empty 'system_prompt'. "
                f"The key '{self.system_prompt_key}' did not yield a valid prompt. Aborting."
            )
            exit(1)
        logging.info(f"[{self.__class__.__name__}] Loaded system_prompt from key: '{self.system_prompt_key}'")

        self._reset_tool_usage_tracking()

    @property
    def memory(self):
        if not hasattr(self._thread_local, "memory"):
            self._thread_local.memory = HybridMemory()
        return self._thread_local.memory

    def _reset_tool_usage_tracking(self) -> None:
        self._tool_usage_summary = {
            "total_counts": {name: 0 for name in ("execute_step", "expand", "fold", "final_answer")},
            "examples_with_tool": {name: 0 for name in ("expand", "fold", "final_answer")},
            "examples_processed": 0,
        }
        self._latest_tool_usage_report: Dict[str, Any] | None = None

    @staticmethod
    def _canonical_tool_name(name: str) -> str:
        mapping = {
            "final_answer": "final_answer",
            "answer": "final_answer",
            "execute_step": "execute_step",
            "commit": "execute_step",
            "expand": "expand",
            "fold": "fold",
            "commit_step": "execute_step",
            "expand_step": "expand",
            "fold_step": "fold",
        }
        return mapping.get(name, name)

    def _collect_tool_usage_from_history(self, history: Any) -> Dict[str, Any]:
        counts = {name: 0 for name in ("execute_step", "expand", "fold", "final_answer")}
        tools_used = set()

        if not isinstance(history, list):
            return {"call_counts": counts, "tools_used": tools_used}

        for entry in history:
            if not isinstance(entry, dict):
                continue
            for call in entry.get("tool_calls", []) or []:
                if not isinstance(call, dict):
                    continue
                raw_name = call.get("tool_name") or call.get("name") or call.get("function", {}).get("name")
                if not raw_name:
                    continue
                tool_name = self._canonical_tool_name(raw_name)
                if tool_name not in counts:
                    continue
                counts[tool_name] += 1
                tools_used.add(tool_name)
        logging.info("[Tool Usage] Collected tool usage from history: %s", counts)
        return {"call_counts": counts, "tools_used": tools_used}

    def _aggregate_tool_usage_for_example(self, histories: List[Any]) -> Dict[str, Any]:
        example_counts = {name: 0 for name in ("execute_step", "expand", "fold", "final_answer")}
        example_tools_used = set()

        for history in histories:
            stats = self._collect_tool_usage_from_history(history)
            for name, value in stats["call_counts"].items():
                example_counts[name] += value
            example_tools_used.update(stats["tools_used"])

        return {"call_counts": example_counts, "tools_used": example_tools_used}

    def _update_tool_usage_summary(self, histories: List[Any]) -> Dict[str, Any]:
        example_stats = self._aggregate_tool_usage_for_example(histories)

        for name, value in example_stats["call_counts"].items():
            self._tool_usage_summary["total_counts"][name] += value

        for name in ("expand", "fold", "final_answer"):
            if name in example_stats["tools_used"]:
                self._tool_usage_summary["examples_with_tool"][name] += 1

        self._tool_usage_summary["examples_processed"] += 1
        return example_stats

    def _compute_tool_usage_report(self) -> Dict[str, Any]:
        processed = self._tool_usage_summary.get("examples_processed", 0)
        total_counts = dict(self._tool_usage_summary.get("total_counts", {}))

        if processed > 0:
            averages = {name: total_counts.get(name, 0) / processed for name in total_counts}
        else:
            averages = {name: 0 for name in total_counts}

        return {
            "examples_processed": processed,
            "total_counts": total_counts,
            "average_per_example": averages,
            "examples_with_tool": dict(self._tool_usage_summary.get("examples_with_tool", {})),
        }

    def _report_tool_usage_summary(self) -> None:
        report = self._compute_tool_usage_report()
        self._latest_tool_usage_report = report
        if report["examples_processed"] == 0:
            logging.info("[Tool Stats] No examples processed; skipping tool usage summary.")
            return

        printable_totals = {
            "execute_step": report["total_counts"].get("execute_step", 0),
            "expand": report["total_counts"].get("expand", 0),
            "fold": report["total_counts"].get("fold", 0),
            "answer": report["total_counts"].get("final_answer", 0),
        }
        printable_averages = {
            "execute_step": report["average_per_example"].get("execute_step", 0.0),
            "expand": report["average_per_example"].get("expand", 0.0),
            "fold": report["average_per_example"].get("fold", 0.0),
            "answer": report["average_per_example"].get("final_answer", 0.0),
        }
        printable_example_counts = {
            "expand": report["examples_with_tool"].get("expand", 0),
            "fold": report["examples_with_tool"].get("fold", 0),
            "answer": report["examples_with_tool"].get("final_answer", 0),
        }

        logging.info("[Tool Stats] Total tool calls: %s", printable_totals)
        logging.info("[Tool Stats] Average calls per example: %s", printable_averages)
        logging.info("[Tool Stats] Examples using expand/fold/answer: %s", printable_example_counts)

    @abstractmethod
    def generate_responses(self, max_workers: int):
        raise NotImplementedError

    @abstractmethod
    def evaluate_responses(self, max_workers: int):
        raise NotImplementedError

    def eval(self):
        self.generate_responses(self.max_workers)
        self.evaluate_responses(self.max_workers)

    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]

        if self.debug:
            data = data[:self.debug_num]
            logging.warning(f"DEBUG MODE: Running only on the first {self.debug_num} samples.")

        return data[self.bidx : self.eidx] if self.eidx != -1 else data[self.bidx :]
