# core/tasks/general_reasoning.py

import copy
import json
import logging
logger = logging.getLogger(__name__)
import re
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .basedataset import BaseDataset
from ..prompts.eval_prompts import  EVALUATION_PROMPT
from ..utils.general_reasoning_utils import extract_last_boxed_content, is_equiv
from ..llm_client import LLMClient
from ..prompts.reason import PROMPT, summarizer_prompt_reason_template

import os

class GeneralReasoningDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_repeat = 1
        self._default_extra_body = {
            "google": {
                "thinking_config": {
                    "include_thoughts": True,
                    # "thinking_budget": 2048     
                },
                "thought_tag_marker": "think"
            }
        }

        self.summarizer_chatter = LLMClient(
            model_name=os.environ.get("SUMMARY_MODEL_NAME", ""), 
            base_url=os.environ.get("SUMMARY_URL", ""),
            api_key=os.environ.get("SUMMARY_API_KEY", ""),
            temperature=0.7,
            max_tokens=8192,
        )

    def _process_single_example(self, example):
        """
        """
        outputs, histories, model_answers = [], [], []
        
        current_problem = example['problem'].replace("Return your final response within \\boxed{}.", "").strip()
        
        for i in range(self.n_repeat):
            max_retries = 5
            attempt = 0
            success = False
            
            while attempt < max_retries:
                try:
                    initial_history = [{"role": "user", "content": PROMPT.format(problem=current_problem)}]
                    
                    self.memory.reset()

                    if self.base_notool_mode:
                        final_reply, _, new_history = self._chat_once(initial_history, self._default_extra_body)
                    else:
                        final_reply, _, new_history = self._chat_multiturn(initial_history, self._default_extra_body)
                    
                    model_answer = extract_last_boxed_content(final_reply)

                    if model_answer is None or not model_answer.strip():
                        raise ValueError("No valid \\boxed{} answer found.")
                    
                    outputs.append(final_reply)
                    histories.append(new_history)
                    model_answers.append(model_answer)
                    success = True
                    break 

                except Exception as e:
                    attempt += 1
                    if attempt >= max_retries:
                        logger.error(f"ID {example['id']} (Rep {i+1}) failed: {e}")
            
            if not success:
                outputs.append("")
                histories.append([])
                model_answers.append("FAILED_GENERATION")

        if any(ans != "FAILED_GENERATION" for ans in model_answers):
            record = example.copy()
            record.update({
                "model_outputs": outputs, 
                "model_histories": histories,
                "model_answers": model_answers
            })
            return record
        return None

    def generate_responses(self, max_workers=10):
        all_examples = self.load_data()
        
        completed_ids = set()
        if os.path.exists(self.raw_output_file):
            with open(self.raw_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        completed_ids.add(data['id'])
                    except Exception:
                        pass

        examples_to_run = [ex for ex in all_examples if ex['id'] not in completed_ids]
        if not examples_to_run:
            logger.info("All examples processed.")
            return

        logger.info(f"Starting generation with {max_workers} threads for {len(examples_to_run)} examples.")

        with open(self.raw_output_file, 'a', encoding='utf-8') as raw_f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {
                    executor.submit(self._process_single_example, ex): ex['id'] 
                    for ex in examples_to_run
                }

                for future in tqdm(as_completed(future_to_id), total=len(examples_to_run), desc="Parallel Generation"):
                    ex_id = future_to_id[future]
                    try:
                        result = future.result()
                        if result:
                            raw_f.write(json.dumps(result) + '\n')
                            raw_f.flush()
                    except Exception as e:
                        logger.error(f"Top-level error for Example ID {ex_id}: {e}")


    def _evaluate_single_case(self, example_idx, example, rep_i):
            """
            """
            pred = example["model_answers"][rep_i] if rep_i < len(example["model_answers"]) else ""
            gt = str(example.get("answer", ""))
            question = example.get("problem", "")
            
            result = {
                "example_idx": example_idx,
                "rep_i": rep_i,
                "pred": pred,
                "gt": gt,
                "is_correct": False,
                "judged_by": "none",
                "new_ans": None,
                "reasoning": None,
                "log_msg": ""
            }

            try:
                if is_equiv(gt, pred):
                    result["is_correct"] = True
                    result["judged_by"] = "is_equiv"
                    result["new_ans"] = gt
                    result["log_msg"] = f"[ID: {example.get('id', '?')}] gt: {gt} | pred: {pred} => Correct by is_equiv"
                else:
                    is_correct_llm, std_ans_llm, reasoning_llm = self.llm_is_equiv(question, pred, gt)
                    result["reasoning"] = reasoning_llm
                    
                    if is_correct_llm:
                        result["is_correct"] = True
                        result["judged_by"] = "llm_is_equiv"
                        result["new_ans"] = std_ans_llm
                        result["log_msg"] = (f"[ID: {example.get('id', '?')}] gt: {gt} | pred: {pred} => Correct by llm_is_equiv\n"
                                            f"    -> Standardized to: {std_ans_llm}")
                    else:
                        result["is_correct"] = False
                        result["judged_by"] = "none"
                        result["log_msg"] = (f"[ID: {example.get('id', '?')}] gt: {gt} | pred: {pred} => Incorrect\n"
                                            f"    -> LLM Reason: {str(reasoning_llm)[:100]}...")
            except Exception as e:
                logger.warning(f"Error evaluating example {example.get('id', '?')} in repetition {rep_i+1}: {e}")
                result["error"] = str(e)

            return result

    def evaluate_responses(self, max_workers=20):
            """
            max_workers: 
            """
            if not os.path.exists(self.raw_output_file):
                logger.error(f"Raw output file {self.raw_output_file} not found. Nothing to evaluate.")
                return

            examples = []
            with open(self.raw_output_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line: continue
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping corrupted line {line_num}: {e}")
                        continue

            num_questions = len(examples)
            if num_questions == 0:
                logger.error("No valid JSON lines found.")
                return

            run_stats_map = {i: 0 for i in range(self.n_repeat)} 

            total_tasks = self.n_repeat * num_questions
            
            logger.info(f"Starting evaluation with {max_workers} workers. Total tasks: {total_tasks}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in range(self.n_repeat):
                    for idx, example in enumerate(examples):
                        futures.append(executor.submit(self._evaluate_single_case, idx, example, i))

                for future in tqdm(as_completed(futures), total=total_tasks, desc="Evaluating"):
                    try:
                        res = future.result()
                        
                        if "error" in res:
                            continue
                            
                        idx = res["example_idx"]
                        rep_i = res["rep_i"]
                        

                        if res["is_correct"]:
                            run_stats_map[rep_i] += 1
                        
                        
                        
                        target_example = examples[idx]
                        
                        if res["new_ans"] is not None:
                            target_example["model_answers"][rep_i] = res["new_ans"]
                        
                        target_example.update({
                            "is_correct": res["is_correct"],
                            "judged_by": res["judged_by"]
                        })
                        
                        if res["reasoning"]:
                            target_example["llm_eval_reasoning"] = res["reasoning"]

                    except Exception as e:
                        logger.error(f"Critical error in result processing: {e}")

            all_run_stats = []
            for i in range(self.n_repeat):
                num_solved = run_stats_map[i]
                all_run_stats.append({
                    "repetition": i + 1,
                    "num_solved": num_solved,
                    "accuracy": num_solved / num_questions if num_questions > 0 else 0,
                })

            final_report = {
                "task_name": self.task_name,
                "num_total": num_questions,
                "accuracy_avg": np.mean([res["accuracy"] for res in all_run_stats]),
                "run_stats": all_run_stats,
                "examples": examples
            }

            with open(self.final_report_file, "w", encoding='utf-8') as f:
                json.dump(final_report, f, indent=4, ensure_ascii=False)
            
            logger.info(f"GeneralReasoning evaluation complete. Report saved to {self.final_report_file}")
    
    def _chat_once(self, messages: List[Dict], extra_body: Dict) -> tuple:
        messages_with_system = []
        if self.system_prompt:
            messages_with_system.append({
                "role": "system",
                "content": self.system_prompt
            })
        messages_with_system.extend(messages)
        logger.info(f"Calling model with messages: {messages_with_system}")
        msg = self.chatter.call_llm(
            messages=messages_with_system,
            extra_body=extra_body,
        )
        logger.info("Model response received.")
        final_messages = messages_with_system + [msg.model_dump(exclude_none=True)]
        
        reason = getattr(msg, "reasoning_content", None)
        return msg.content, (reason if self.enable_thinking else None), final_messages


    def _chat_multiturn(self, messages: List[Dict], extra_body: Dict) -> tuple:
        if any(tool['function']['name'] == 'record_step' for tool in self.tools):
            logger.info("[GeneralReasoning] Using Single Mode with Record Step tool.")
            return self._chat_multiturn_single_mode_record(messages, extra_body)

        if any(tool['function']['name'] == 'commit' for tool in self.tools):
            logger.info("[GeneralReasoning] Using Single Mode with Scratchpad tools.")
            return self._chat_multiturn_single_mode_commit(messages, extra_body)

        logger.info("[GeneralReasoning] Using Single Mode.")
        return self._chat_multiturn_single_mode(messages, extra_body)

    def _chat_multiturn_single_mode_record(self, messages: List[Dict], extra_body: Dict) -> tuple:
        """
        Single : （record_step, expand, fold, final_answer）
        """
        interaction_log = []
        initial_messages = list(messages)
        prompt_messages: List[Dict[str, Any]] = list(messages)
        
        currently_expanded_ids = set()

        round_num = 1
        no_tool_retry_count = 0
        MAX_NO_TOOL_RETRIES = 3
        MAX_ROUNDS = 30

        while round_num <= MAX_ROUNDS:
            logger.info(f"[Single Mode] Round {round_num}")
            
            current_interaction = {
                "round": round_num, "model_type": "single", "input_messages": [], "output": {},
                "tool_calls": [], "tool_results": [], "notes": [], "next_step_prefix": "",
            }
            
            messages_with_system = []
            if self.system_prompt:
                messages_with_system.append({"role": "system", "content": self.system_prompt})
            messages_with_system.extend(prompt_messages)
            current_interaction["input_messages"] = [m.copy() for m in messages_with_system]
            
            logger.info(f"[Single Mode] Calling model for round {round_num}...")
            msg = self.chatter.call_llm(messages=messages_with_system, tools=self.tools, extra_body=extra_body)
            
            logger.info("[Single Mode] Raw response received")
            reason = getattr(msg, "reasoning_content", None)
            msg_dict = msg.model_dump(exclude_none=True)
            current_interaction["output"] = {
                "content": msg_dict.get("content"),
                "reasoning": reason if self.enable_thinking else None,
                "raw_response": msg_dict
            }
            
            tool_calls = msg_dict.get("tool_calls")
            if not tool_calls:
                current_interaction["notes"].append("No tool calls in this round")
                interaction_log.append(current_interaction)
                no_tool_retry_count += 1
                if no_tool_retry_count >= MAX_NO_TOOL_RETRIES:
                    return msg.content, (reason if self.enable_thinking else None), interaction_log
                prompt_messages = self._rebuild_prompt_messages_single_record(initial_messages, interaction_log, currently_expanded_ids)
                if no_tool_retry_count < MAX_NO_TOOL_RETRIES - 1:
                    prompt_messages.append({"role": "user", "content": "You must call a function. What is the NEXT step?"})
                # output final answer
                if no_tool_retry_count == MAX_NO_TOOL_RETRIES:
                    prompt_messages.append({"role": "user", "content": "Please provide the FINAL ANSWER now using the final_answer function."})
                round_num += 1
                continue
            
            no_tool_retry_count = 0
            
            executed_any = False
            final_answer_content = None
            
            for call in tool_calls:
                name, call_id = call['function']['name'], call['id']
                try:
                    args = json.loads(call['function']['arguments'])
                except json.JSONDecodeError:
                    logger.warning(f"[Single Mode] Failed to decode arguments for tool {name}")
                    continue
                
                tool_call_record = {"tool_name": name, "arguments": args, "call_id": call_id}
                
                logger.info(f"[Single Mode] Executing tool: {name} with args: {args}")
                result_content, next_step_prefix = "", ""
                
                if name == "record_step":
                    detailed_step = args.get("detailed_step")
                    
                    if not detailed_step:
                        result_content, tool_call_record["status"] = "Error: 'detailed_step' parameter is required.", "error"
                    else:
                        logger.info("[Single Mode] Generating summary with dedicated summarizer...")
                        summarizer_prompt = summarizer_prompt_reason_template.format(observation=detailed_step)
                        summary_msg = self.summarizer_chatter.call_llm(messages=[{"role": "user", "content": summarizer_prompt}])
                        summary_text = summary_msg.content.strip()
                        logger.info(f"[Single Mode] Generated Summary: {summary_text}")

                        step_id = self.memory.end_step(
                            detailed_reasoning=detailed_step,
                            summary_text=summary_text,
                            reasoning_content_from_msg=reason
                        )
                        
                        result_content = f"[Step ID: {step_id}] {summary_text}"
                        executed_any, tool_call_record["status"] = True, "success"
                        tool_call_record["step_id"] = step_id
                        next_step_prefix = f"Step {step_id} completed. "
                
                elif name == "expand":
                    step_id = str(args.get("step_id", ""))
                    summary_data = self.memory.get_summary_data(step_id)
                    
                    if not summary_data:
                        result_content, tool_call_record["status"] = f"Error: Step with ID '{step_id}' not found.", "error"
                    else:
                        result_content = f"[Expanded Step ID: {step_id}] {summary_data['original']}"
                        currently_expanded_ids.add(step_id)
                        executed_any, tool_call_record["status"] = True, "success"
                        tool_call_record["expanded_content"] = summary_data['original']
                        next_step_prefix = f"Step {step_id} has been expanded. "
                        
                elif name == "fold":
                    step_id = str(args.get("step_id", ""))
                    summary_data = self.memory.get_summary_data(step_id)
                    
                    if not summary_data:
                        result_content, tool_call_record["status"] = f"Error: Step with ID '{step_id}' not found.", "error"
                    else:
                        result_content = f"[Folded Step ID: {step_id}] {summary_data['summary']}"
                        currently_expanded_ids.discard(step_id)
                        executed_any, tool_call_record["status"] = True, "success"
                        tool_call_record["folded_to"] = summary_data['summary']
                        next_step_prefix = f"Step {step_id} has been folded. "
                        
                elif name == "final_answer":
                    answer_text = args.get("final_answer")
                    if not answer_text:
                        result_content, tool_call_record["status"] = "Error: 'final_answer' parameter is required.", "error"
                    else:
                        final_answer_content, result_content = answer_text, "Final answer recorded."
                        executed_any, tool_call_record["status"] = True, "success"
                        tool_call_record["answer"] = final_answer_content 
        
                else:
                    result_content, tool_call_record["status"] = f"Error: Unknown tool '{name}'.", "error"
                
                tool_call_record["result"] = result_content
                current_interaction["tool_calls"].append(tool_call_record)
                current_interaction["tool_results"].append({"role": "tool", "tool_call_id": call_id, "content": result_content})
                current_interaction["next_step_prefix"] = next_step_prefix

            if not executed_any:
                current_interaction["notes"].append("No tools successfully executed")
                interaction_log.append(current_interaction)
                logger.warning("[Single Mode] No tools successfully executed. Halting.")
                return msg.content, (reason if self.enable_thinking else None), interaction_log
            
            interaction_log.append(current_interaction)
            
            if final_answer_content:
                logger.info("[Single Mode] final_answer called. Ending.")
                return final_answer_content, (reason if self.enable_thinking else None), interaction_log
            
            prompt_messages = self._rebuild_prompt_messages_single_record(initial_messages, interaction_log, currently_expanded_ids)
            
            if not prompt_messages or "What is the NEXT step?" not in prompt_messages[-1].get("content", ""):
                prompt_messages.append({"role": "user", "content": next_step_prefix + "What is the NEXT step?"})

            round_num += 1
        
        logger.warning(f"[Single Mode] Exceeded max rounds ({MAX_ROUNDS})")
        last_content = interaction_log[-1]["output"].get("content", "") if interaction_log else ""
        return last_content, None, interaction_log 

    def _chat_multiturn_single_mode(self, messages: List[Dict], extra_body: Dict) -> tuple:
        """
        Single ：（execute_step, expand, fold, final_answer）
        """
        interaction_log = []
        initial_messages = list(messages)
        prompt_messages: List[Dict[str, Any]] = list(messages)
        
        currently_expanded_ids = set()  

        round_num = 1
        no_tool_retry_count = 0
        MAX_NO_TOOL_RETRIES = 3
        MAX_ROUNDS = 30

        while round_num <= MAX_ROUNDS:
            logger.info(f"[Single Mode] Round {round_num}")
            
            current_interaction = {
                "round": round_num,
                "model_type": "single",
                "input_messages": [],
                "output": {},
                "tool_calls": [],
                "tool_results": [],
                "notes": [],
                "next_step_prefix": "",
            }
            
            messages_with_system = []
            if self.system_prompt:
                messages_with_system.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            messages_with_system.extend(prompt_messages)
            
            current_interaction["input_messages"] = [m.copy() for m in messages_with_system]
            
            logger.info(f"[Single Mode] Calling model for round {round_num}...")
            logger.info(f"[Single Mode] Prompt messages: {messages_with_system}")

            msg = self.chatter.call_llm(
                messages=messages_with_system,
                tools=self.tools,
                extra_body=extra_body
            )
            
            logger.info("[Single Mode] Raw response received")
            logger.info(json.dumps(msg.model_dump(), ensure_ascii=False, indent=2))
            logger.info(f"[Single Mode] Raw response received: {msg.model_dump()}")
            reason = getattr(msg, "reasoning_content", None)
            msg_dict = msg.model_dump(exclude_none=True)
            
            current_interaction["output"] = {
                "content": msg_dict.get("content"),
                "reasoning": reason if self.enable_thinking else None,
                "raw_response": msg_dict
            }
            
            tool_calls = msg_dict.get("tool_calls")
            
            if not tool_calls:
                current_interaction["notes"].append("No tool calls in this round")
                logger.info("[Single Mode] No tool calls detected")
                interaction_log.append(current_interaction)
                
                no_tool_retry_count += 1
                current_interaction["notes"].append(f"No tool call, retry {no_tool_retry_count}/{MAX_NO_TOOL_RETRIES}")
                
                if no_tool_retry_count >= MAX_NO_TOOL_RETRIES:
                    logger.info("[Single Mode] No tool calls after 3 retries. Ending.")
                    return msg.content, (reason if self.enable_thinking else None), interaction_log
                
                prompt_messages = self._rebuild_prompt_messages_single(
                    initial_messages, 
                    interaction_log, 
                    currently_expanded_ids
                )
                prompt_messages.append({
                    "role": "user",
                    "content": "You must call a function. What is the NEXT step?"
                })
                round_num += 1
                continue
            
            no_tool_retry_count = 0
            
            executed_any = False
            final_answer_content = None
            
            for call in tool_calls:
                name = call['function']['name']
                call_id = call['id']
                
                try:
                    args = json.loads(call['function']['arguments'])
                except json.JSONDecodeError:
                    logger.warning(f"[Single Mode] Failed to decode arguments for tool {name}")
                    continue
                
                tool_call_record = {
                    "tool_name": name,
                    "arguments": args,
                    "call_id": call_id
                }
                
                logger.info(f"[Single Mode] Executing tool: {name}")
                logger.info(f"[Single Mode] Executing tool: {name}")
                logger.info(f"[Single Mode] Tool arguments: {args}")
                logger.info(f"[Single Mode] Tool arguments: {args}")
                result_content = ""
                next_step_prefix = ""
                if name == "execute_step":
                    detailed_reasoning = args.get("detailed_reasoning")
                    summary_text = args.get("summary_text")
                    
                    if not detailed_reasoning or not summary_text:
                        result_content = "Error: 'detailed_reasoning' and 'summary_text' are required."
                        tool_call_record["status"] = "error"
                    else:
                        summary_id = self.memory.end_step(
                            detailed_reasoning=detailed_reasoning,
                            summary_text=summary_text,
                            reasoning_content_from_msg=reason
                        )
                        result_content = f"[Summary ID: {summary_id}] {summary_text}"
                        executed_any = True
                        next_step_prefix = f"Step {summary_id} completed. "
                        tool_call_record["status"] = "success"
                        tool_call_record["summary_id"] = summary_id

                elif name == "expand":
                    summary_id = str(args.get("summary_id",""))
                    summary_data = self.memory.get_summary_data(summary_id)
                    
                    if not summary_data:
                        result_content = f"Error: Summary with ID '{summary_id}' not found."
                        tool_call_record["status"] = "error"
                    else:
                        result_content = f"[Expanded Summary ID: {summary_id}] {summary_data['original']}"
                        currently_expanded_ids.add(summary_id)
                        executed_any = True
                        next_step_prefix = f"Summary {summary_id} has been expanded. "
                        tool_call_record["status"] = "success"
                        tool_call_record["expanded_content"] = summary_data['original']
                        
                elif name == "fold":
                    summary_id = str(args.get("summary_id"))
                    summary_data = self.memory.get_summary_data(summary_id)
                    
                    if not summary_data:
                        result_content = f"Error: Summary with ID '{summary_id}' not found."
                        tool_call_record["status"] = "error"
                    else:
                        result_content = f"[Folded Summary ID: {summary_id}] {summary_data['summary']}"
                        currently_expanded_ids.discard(summary_id)
                        executed_any = True
                        next_step_prefix = f"Summary {summary_id} has been folded. "
                        tool_call_record["status"] = "success"
                        tool_call_record["folded_to"] = summary_data['summary']
                        
                elif name == "final_answer":
                    answer_text = args.get("final_answer")
                    
                    if not answer_text:
                        result_content = "Error: 'final_answer' parameter is required."
                        tool_call_record["status"] = "error"
                    else:
                        answer_text = answer_text.replace('\x08oxed{', r'\boxed{')
                        final_answer_content = answer_text
                        result_content = "Final answer recorded."
                        executed_any = True
                        tool_call_record["status"] = "success"
                        tool_call_record["answer"] = answer_text
                else:
                    result_content = f"Error: Unknown tool '{name}'."
                    tool_call_record["status"] = "error"
                
                tool_call_record["result"] = result_content


                current_interaction["tool_calls"].append(tool_call_record)
                
                current_interaction["tool_results"].append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_content
                })
                current_interaction["next_step_prefix"] = next_step_prefix

            
            if not executed_any:
                current_interaction["notes"].append("No tools successfully executed")
                interaction_log.append(current_interaction)
                logger.warning("[Single Mode] No tools successfully executed. Halting.")
                return msg.content, (reason if self.enable_thinking else None), interaction_log
            
            interaction_log.append(current_interaction)
            logger.info("[Single Mode] current interaction logged. ")
            logger.info(f"[Single Mode] {json.dumps(current_interaction, ensure_ascii=False, indent=2)}")
            if final_answer_content:
                logger.info("[Single Mode] final_answer called. Ending.")
                return final_answer_content, (reason if self.enable_thinking else None), interaction_log
            
            prompt_messages = self._rebuild_prompt_messages_single(
                initial_messages, 
                interaction_log, 
                currently_expanded_ids
            )
            
            if not prompt_messages or "What is the NEXT step?" not in prompt_messages[-1].get("content", ""):
                prompt_messages.append({
                    "role": "user",
                    "content": next_step_prefix + "What is the NEXT step?"
                })

            round_num += 1
        
        logger.warning(f"[Single Mode] Exceeded max rounds ({MAX_ROUNDS})")
        last_content = interaction_log[-1]["output"].get("content", "") if interaction_log else ""
        return last_content, None, interaction_log

    def _chat_multiturn_single_mode_commit(self, messages: List[Dict], extra_body: Dict) -> tuple:
        """
        Single ：（commit, expand, fold, final_answer）
        """
        interaction_log = []
        initial_messages = list(messages)
        prompt_messages: List[Dict[str, Any]] = list(messages)
        
        currently_expanded_ids = set()  

        round_num = 1
        no_tool_retry_count = 0
        MAX_NO_TOOL_RETRIES = 3
        MAX_ROUNDS = 30

        logger.info("Starting Single Mode with Commit Step tool...")
        logger.info(f"Tools available: {[tool['function']['name'] for tool in self.tools]}")

        while round_num <= MAX_ROUNDS:
            logger.info(f"[Single Mode] Round {round_num}")
            
            next_step_prefix_accumulated = {}
            current_interaction = {
                "round": round_num,
                "model_type": "single",
                "input_messages": [],
                "output": {},
                "tool_calls": [],
                "tool_results": [],
                "notes": [],
                "next_step_prefix": "",
                "next_step_prefix_accumulated": next_step_prefix_accumulated, 
            }
            
            messages_with_system = []
            if self.system_prompt:
                messages_with_system.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            if not self.system_prompt:
                logger.info("[Single Mode] No system prompt provided.")
                # exit()
            messages_with_system.extend(prompt_messages)
            
            current_interaction["input_messages"] = [m.copy() for m in messages_with_system]
            
            logger.info(f"[Single Mode] Calling model for round {round_num}...")
            logger.info(f"[Single Mode] Prompt messages: {messages_with_system}")

            msg = self.chatter.call_llm(
                messages=messages_with_system,
                tools=self.tools,
                extra_body=extra_body
            )
            
            logger.info("[Single Mode] Raw response received")
            logger.info(json.dumps(msg.model_dump(), ensure_ascii=False, indent=2))
            logger.info(f"[Single Mode] Raw response received: {msg.model_dump()}")
            reason = getattr(msg, "reasoning_content", None)
            msg_dict = msg.model_dump(exclude_none=True)
            
            current_interaction["output"] = {
                "content": msg_dict.get("content"),
                "reasoning": reason if self.enable_thinking else None,
                "raw_response": msg_dict
            }
            
            tool_calls = msg_dict.get("tool_calls")
            
            if not tool_calls:
                current_interaction["notes"].append("No tool calls in this round")
                logger.info("[Single Mode] No tool calls detected")
                interaction_log.append(current_interaction)
                
                no_tool_retry_count += 1
                current_interaction["notes"].append(f"No toxwol call, retry {no_tool_retry_count}/{MAX_NO_TOOL_RETRIES}")
                
                if no_tool_retry_count > MAX_NO_TOOL_RETRIES:
                    logger.info("[Single Mode] No tool calls after 3 retries. Ending.")
                    return msg.content, (reason if self.enable_thinking else None), interaction_log
                                
                prompt_messages = self._rebuild_prompt_messages_single_commit(
                    initial_messages, 
                    interaction_log, 
                    currently_expanded_ids
                )

                if no_tool_retry_count == MAX_NO_TOOL_RETRIES:
                    prompt_messages.append({
                        "role": "user",
                        "content": "Please provide the FINAL ANSWER now using the final_answer function."
                    })
                
                if no_tool_retry_count < MAX_NO_TOOL_RETRIES:
                    available_tool_names = [tool['function']['name'] for tool in self.tools]
                    formatted_tools = ", ".join(available_tool_names)
                    if len(current_interaction["input_messages"]) > 2: # exclude system prompt and first user message
                        last_round_message = current_interaction["input_messages"][-1].get("content", "")
                        feedback_content = f"{last_round_message} You must call a function to proceed. Choose one of the available tools: {formatted_tools}."
                    else:
                        feedback_content = f"You must call a function to proceed. Choose one of the available tools: {formatted_tools}."
                    prompt_messages.append({
                        "role": "user",
                        "content": feedback_content
                    })
                round_num += 1
                continue
            
            no_tool_retry_count = 0
            
            executed_any = False
            final_answer_content = None
            
            for call in tool_calls:
                name = call['function']['name']
                call_id = call['id']
                
                try:
                    args = json.loads(call['function']['arguments'])
                except json.JSONDecodeError:
                    logger.warning(f"[Single Mode] Failed to decode arguments for tool {name}")
                    continue
                
                tool_call_record = {
                    "tool_name": name,
                    "arguments": args,
                    "call_id": call_id
                }
                
                logger.info(f"[Single Mode] Executing tool: {name}")
                logger.info(f"[Single Mode] Executing tool: {name}")
                logger.info(f"[Single Mode] Tool arguments: {args}")
                logger.info(f"[Single Mode] Tool arguments: {args}")
                result_content = ""
                next_step_prefix = ""

                if name == "commit":  
                    detailed_reasoning = args.get("detailed_reasoning")
                    summary = args.get("summary") 
                    
                    if not detailed_reasoning or not summary:
                        result_content = "Error: 'detailed_reasoning' and 'summary' are required." 
                        tool_call_record["status"] = "error"
                    else:
                        step_id = self.memory.end_step(
                            detailed_reasoning=detailed_reasoning,
                            summary_text=summary, 
                            reasoning_content_from_msg=reason
                        )
                        result_content = f"[Step ID: {step_id}] {summary}" 
                        executed_any = True
                        next_step_prefix = f"Step {step_id} committed. " 
                        tool_call_record["status"] = "success"
                        tool_call_record["step_id"] = step_id 

                elif name == "expand":
                    step_id = str(args.get("step_id",""))  
                    reason = args.get("reason", "")
                    reasoning_content = getattr(msg, "reasoning_content", None)
                    summary_data = self.memory.get_summary_data(step_id)  
                    
                    if not summary_data:
                        result_content = f"Error: Step with ID '{step_id}' not found."  
                        tool_call_record["status"] = "error"
                    else:
                        result_content = f"[Expanded Step ID: {step_id}] {summary_data['original']}" 
                        currently_expanded_ids.add(step_id)
                        executed_any = True
                        next_step_prefix = f"{reason} Step {step_id} has been expanded for that purpose. You should use the detailed information to commit a new step. "
                        # next_step_prefix = f"{reasoning_content} Step {step_id} has been expanded now. "
                        next_step_prefix_accumulated[step_id] = next_step_prefix                      
                        tool_call_record["status"] = "success"
                        tool_call_record["expanded_content"] = summary_data['original']
                        
                elif name == "fold":
                    step_id = str(args.get("step_id"))  
                    reason = args.get("reason", "")
                    reasoning_content = getattr(msg, "reasoning_content", None)
                    summary_data = self.memory.get_summary_data(step_id)  
                    
                    if not summary_data:
                        result_content = f"Error: Step with ID '{step_id}' not found."  
                        tool_call_record["status"] = "error"
                    else:
                        result_content = f"[Folded Step ID: {step_id}] {summary_data['summary']}" # 
                        currently_expanded_ids.discard(step_id)
                        executed_any = True
                        next_step_prefix = f"{reason} Step {step_id} has been folded since its detailed view is no longer needed. You should keep commiting further steps or provide the final answer. "  
                        # next_step_prefix = f"{reasoning_content} Step {step_id} has been folded now. "
                        next_step_prefix_accumulated.pop(step_id, None)
                        tool_call_record["status"] = "success"
                        tool_call_record["folded_to"] = summary_data['summary']
                        
                elif name == "final_answer":
                    answer_text = args.get("final_answer")
                    
                    if not answer_text:
                        result_content = "Error: 'final_answer' parameter is required."
                        tool_call_record["status"] = "error"
                    else:
                        answer_text = answer_text.replace('\x08oxed{', r'\boxed{')
                        final_answer_content = answer_text
                        result_content = "Final answer recorded."
                        executed_any = True
                        tool_call_record["status"] = "success"
                        tool_call_record["answer"] = answer_text
                else:
                    result_content = f"Error: Unknown tool '{name}'."
                    tool_call_record["status"] = "error"
                
                tool_call_record["result"] = result_content


                current_interaction["tool_calls"].append(tool_call_record)
                
                current_interaction["tool_results"].append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_content
                })
                current_interaction["next_step_prefix"] = next_step_prefix
                current_interaction["next_step_prefix_accumulated"] = next_step_prefix_accumulated

            
            if not executed_any:
                current_interaction["notes"].append("No tools successfully executed")
                interaction_log.append(current_interaction)
                logger.warning("[Single Mode] No tools successfully executed. Halting.")
                return msg.content, (reason if self.enable_thinking else None), interaction_log
            
            interaction_log.append(current_interaction)
            logger.info("[Single Mode] current interaction logged. ")
            logger.info(f"[Single Mode] {json.dumps(current_interaction, ensure_ascii=False, indent=2)}")

            if final_answer_content:
                logger.info("[Single Mode] final_answer called. Ending.")
                return final_answer_content, (reason if self.enable_thinking else None), interaction_log
            
            prompt_messages = self._rebuild_prompt_messages_single_commit(
                initial_messages, 
                interaction_log, 
                currently_expanded_ids
            )
            
            if self.accumulated_next_step_prefix:
                next_step_prefix = "\n".join(next_step_prefix_accumulated.values()) + "\n"
            if not prompt_messages or "What is the NEXT step?" not in prompt_messages[-1].get("content", ""):
                prompt_messages.append({
                    "role": "user",
                    "content": next_step_prefix + "What is the NEXT step?"
                })

            round_num += 1
        
        logger.warning(f"[Single Mode] Exceeded max rounds ({MAX_ROUNDS})")
        last_content = interaction_log[-1]["output"].get("content", "") if interaction_log else ""
        return last_content, None, interaction_log


    def _rebuild_prompt_messages_single(
        self, 
        initial_messages: List[Dict], 
        interaction_log: List[Dict],
        currently_expanded_ids: set
    ) -> List[Dict]:
        """
        Single ： prompt messages
        """
        prompt_messages = list(initial_messages)
        
        logger.info(f"[Rebuild Single] Starting with {len(initial_messages)} initial messages")
        logger.info(f"[Rebuild Single] Currently expanded IDs: {currently_expanded_ids}")
        
        execute_step_count = 0
        for log_entry in interaction_log:
            for tool_call in log_entry["tool_calls"]:
                if tool_call["tool_name"] == "execute_step" and tool_call["status"] == "success":
                    summary_id = str(tool_call["summary_id"])
                    summary_text = tool_call["arguments"]["summary_text"]
                    
                    if summary_id in currently_expanded_ids:
                        summary_data = self.memory.get_summary_data(summary_id)
                        output_content = f"[Expanded Summary ID: {summary_id}]\n\n{summary_data['original']}"
                        logger.info(f"[Rebuild Single] Added expanded summary {summary_id}")
                    else:
                        output_content = f"[Summary ID: {summary_id}] {summary_text}"
                        logger.info(f"[Rebuild Single] Added folded summary {summary_id}")
                    
                    prompt_messages.append({
                        "role": "assistant",
                        "content": output_content
                    })
                    execute_step_count += 1
                    break
        
        logger.info(f"[Rebuild Single] Added {execute_step_count} execute_step messages")
        logger.info(f"[Rebuild Single] Final prompt has {len(prompt_messages)} messages")
        return prompt_messages

    def _rebuild_prompt_messages_single_commit(
        self, 
        initial_messages: List[Dict], 
        interaction_log: List[Dict],
        currently_expanded_ids: set
    ) -> List[Dict]:
        """
        Single ： prompt messages
        """
        prompt_messages = list(initial_messages)
        
        logger.info(f"[Rebuild Single] Starting with {len(initial_messages)} initial messages")
        logger.info(f"[Rebuild Single] Currently expanded IDs: {currently_expanded_ids}")
        
        commit_count = 0
        for log_entry in interaction_log:
            for tool_call in log_entry["tool_calls"]:
                if tool_call["tool_name"] == "commit" and tool_call["status"] == "success":
                    
                    step_id = str(tool_call["step_id"])
                    
                    summary_text = tool_call["arguments"]["summary"]
                    
                    if step_id in currently_expanded_ids:
                        step_data = self.memory.get_summary_data(step_id)
                        if not step_data:
                            logger.warning(f"[Rebuild Single] Could not find data for step_id {step_id}. Skipping.")
                            continue
                        
                        output_content = f"[Expanded Step ID: {step_id}]\n\n{step_data['original']}"
                        logger.info(f"[Rebuild Single] Added expanded step {step_id}")
                    else:
                        output_content = f"[Step ID: {step_id}] {summary_text}"
                        logger.info(f"[Rebuild Single] Added folded step {step_id}")
                    
                    prompt_messages.append({
                        "role": "assistant",
                        "content": output_content
                    })
                    commit_count += 1
                    break
        
        logger.info(f"[Rebuild Single] Added {commit_count} commit messages")
        logger.info(f"[Rebuild Single] Final prompt has {len(prompt_messages)} messages")
        return prompt_messages


    def _rebuild_prompt_messages_single_record(
            self, 
            initial_messages: List[Dict], 
            interaction_log: List[Dict],
            currently_expanded_ids: set
        ) -> List[Dict]:
            prompt_messages = list(initial_messages)
            
            logger.info(f"[Rebuild Single] Starting with {len(initial_messages)} initial messages")
            logger.info(f"[Rebuild Single] Currently expanded IDs: {currently_expanded_ids}")
            
            record_step_count = 0
            for log_entry in interaction_log:
                for tool_call in log_entry["tool_calls"]:
                    if tool_call["tool_name"] == "record_step" and tool_call["status"] == "success":
                        
                        step_id = str(tool_call["step_id"])
                        
                        step_data = self.memory.get_summary_data(step_id)
                        if not step_data:
                            logger.warning(f"[Rebuild Single] Could not find data for step_id {step_id}. Skipping.")
                            continue

                        summary_text = step_data['summary']
                        original_content = step_data['original']
                        
                        if step_id in currently_expanded_ids:
                            output_content = f"[Expanded Step ID: {step_id}]\n\n{original_content}"
                            logger.info(f"[Rebuild Single] Added expanded step {step_id}")
                        else:
                            output_content = f"[Step ID: {step_id}] {summary_text}"
                            logger.info(f"[Rebuild Single] Added folded step {step_id}")
                        
                        prompt_messages.append({
                            "role": "assistant",
                            "content": output_content
                        })
                        record_step_count += 1
                        break 
            
            logger.info(f"[Rebuild Single] Added {record_step_count} record_step messages")
            logger.info(f"[Rebuild Single] Final prompt has {len(prompt_messages)} messages")
            return prompt_messages


    def llm_is_equiv(self, question: str, predicted_answer: str, ground_truth_answer: str) -> (bool, str, str):
            """
            
            Returns:
                is_correct (bool): 
                standardized_answer (str):  ( predicted_answer)
                reasoning (str): LLM  debug
            """
            predicted_answer = str(predicted_answer) if predicted_answer is not None else ""
            ground_truth_answer = str(ground_truth_answer) if ground_truth_answer is not None else ""

            prompt = EVALUATION_PROMPT.replace("{question}", question) \
                                  .replace("{predicted_answer}", predicted_answer) \
                                  .replace("{ground_truth_answer}", ground_truth_answer)
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
                msg = self.eval_chatter.call_llm(messages=messages)
                full_response = msg.content.strip()
                
                # print(f"--- Eval ---\nQ: {question}\nPred: {predicted_answer}\nGT: {ground_truth_answer}\nLLM: {full_response}\n")

                is_correct = False
                if re.search(r"Correctness:\s*(TRUE|True|true)", full_response):
                    is_correct = True
                
                std_answer_match = re.search(r"Standardized Answer:\s*(.+)", full_response, re.IGNORECASE)
                
                if std_answer_match:
                    standardized_answer = std_answer_match.group(1).strip()
                else:
                    standardized_answer = predicted_answer if is_correct else ground_truth_answer
                    logger.warning(f"Failed to parse Standardized Answer. Using fallback. Response: {full_response}")

                return is_correct, standardized_answer, full_response

            except Exception as e:
                logger.error(f"LLM evaluation failed: {e}")
                return False, predicted_answer, f"Exception: {e}"
