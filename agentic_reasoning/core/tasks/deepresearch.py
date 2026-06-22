# core/tasks/deepresearch.py

import copy
import json
import logging
import os
from typing import Any, Dict, List
import re
import time 
import json5
from tqdm import tqdm
import threading
import datetime
from .basedataset import BaseDataset
from ..prompts.eval_prompts import JUDGE_PROMPT_BROWSECOMP_OFFICIAL,JUDGE_PROMPT_XBENCH
from ..llm_client import LLMClient
import litellm
from transformers import AutoTokenizer
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")



extracted_answer_format_for_confidence = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
            "additionalProperties": False
        },
        "strict": True
    }
}

extracted_answer_format_for_xbench = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "": {"type": "string"},
                "": {"type": "string"},
                "": {"type": "string", "enum": ["", ""]},
            },
            "required": ["", "", ""],
            "additionalProperties": False
        },
        "strict": True
    }
}


class DeepResearchDataset(BaseDataset):
    def __init__(self, task_name, dptool_only=False, react_mode="fn_call",  **kwargs):
        super().__init__(**kwargs)
        self._default_extra_body = {}

        self.task_name = task_name

        
        self.summarizer_chatter = LLMClient(
            model_name=os.environ.get("LOGGER_MODEL_NAME", ""), 
            base_url=os.environ.get("LOGGER_MODEL_URL", ""),
            api_key=os.environ.get("LOGGER_MODEL_API_KEY", ""),
            temperature=0.0,
        )

        # self.dptool_only = dptool_only

        self.react_mode = react_mode        
    
    def _process_single_example(self, example: dict):
        """ worker：， result_record  None"""


        example_id = example.get("id", "N/A")
        logging.info(f"[TID] example_id={example_id}, thread={threading.get_ident()}")
        try:
            self.memory.reset()

            question = example.get("question")
            if not question:
                logging.warning(f"[DeepResearch] Skipping example id={example_id}: missing question")
                return None

            initial_history = [{"role": "user", "content": question}]

            if self.react_mode == "fn_call":
                final_answer, full_interaction_history = self._run_turn_dual_inference_fn_call(initial_history)
            else:
                final_answer, full_interaction_history = self._run_turn_prompt_react_multitool_v1(initial_history)

            result_record = example.copy()
            result_record.update({
                "model_answer": final_answer,
                "interaction_history": full_interaction_history
            })
            return result_record

        except Exception:
            logging.exception(f"[DeepResearch] Error processing example id={example_id}. Skipping.")
            return None


    def generate_responses(self, max_workers: int = 10):
        all_examples = self.load_data()

        completed_ids = set()
        if os.path.exists(self.raw_output_file):
            with open(self.raw_output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            completed_ids.add(data["id"])
                    except Exception:
                        continue

        examples_to_run = [ex for ex in all_examples if ex.get("id") not in completed_ids]
        if not examples_to_run:
            logging.info("[DeepResearch] All examples have been processed.")
            return

        logging.info(f"[DeepResearch] Starting generation with {max_workers} threads for {len(examples_to_run)} examples.")

        with open(self.raw_output_file, "a", encoding="utf-8") as raw_f:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {
                    executor.submit(self._process_single_example, ex): ex.get("id", "N/A")
                    for ex in examples_to_run
                }

                for future in tqdm(as_completed(future_to_id),
                                total=len(future_to_id),
                                desc="DeepResearch Parallel Generation"):
                    ex_id = future_to_id[future]
                    try:
                        result_record = future.result()
                        if result_record:
                            raw_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                            raw_f.flush()
                            logging.info(f"[DeepResearch] Completed example id={ex_id}")
                    except Exception as e:
                        logging.error(f"[DeepResearch] Top-level error for example id={ex_id}: {e}")



    def _run_turn_prompt_react_multitool_v1(self, history: list) -> tuple:
        
        initial_base_messages = []
        if hasattr(self, 'system_prompt') and self.system_prompt:
            initial_base_messages.append({"role": "system", "content": self.system_prompt+today_date()})
        
        user_question = next((h["content"] for h in history if h["role"] == "user"), "")
        if user_question:
            initial_base_messages.append({"role": "user", "content": user_question})
        else:
            logging.error("Could not find user question in initial history.")
            return "Error: No user question found.", []

        interaction_log = []
        round_num = 1
        MAX_ROUNDS = int(os.getenv('MAX_ROUNDS', 100))
        max_tokens = int(os.getenv("MAX_REASON_TOKENS", 108 * 1024))
        start_time = time.time()

        currently_expanded_ids: Dict[str, set] = {"thought": set(), "obs": set()}
        
        last_step_thought_id: Optional[str] = None
        last_step_obs_id: Optional[str] = None
        
        current_thought_id: Optional[str] = None
        current_obs_id: Optional[str] = None

        while round_num <= MAX_ROUNDS:
            if time.time() - start_time > 150 * 60:
                return "Timeout", interaction_log

            round_header = f"--- [Prompt ReAct] Round {round_num} ---"
            logging.info(round_header)

            pending_t_set = set()
            pending_o_set = set()
            if current_thought_id:
                pending_t_set.add(current_thought_id)
            if current_obs_id:
                pending_o_set.add(current_obs_id)

            if last_step_thought_id:
                t_rec = self.memory.get_thought(last_step_thought_id)
                if t_rec and not t_rec.get('summary'):
                    pending_t_set.add(last_step_thought_id)

            if last_step_obs_id:
                o_rec = self.memory.get_observation(last_step_obs_id)
                if o_rec and not o_rec.get('summary'):
                    pending_o_set.add(last_step_obs_id)

            prompt_messages = self._rebuild_prompt_messages_with_pending_v1(
                initial_base_messages,
                self.memory,
                currently_expanded_ids,
                list(pending_t_set), 
                list(pending_o_set)
            )

            current_tokens = self.count_tokens(prompt_messages)
            if current_tokens > max_tokens:
                logging.warning(f"Token limit reached ({current_tokens}). Forcing answer.")
                prompt_messages.append({"role": "user", "content": "Memory limit reached. Output <answer> now."})

            current_interaction = {
                "round": round_num,
                "input_messages": copy.deepcopy(prompt_messages),
                "output": {},
                "tool_calls": [],
                "notes": []
            }

            logging.info(f"[Prompt ReAct] Prompt used: {json.dumps(prompt_messages, ensure_ascii=False, indent=4)}")
            # print(f"[Prompt ReAct] Prompt used: {json.dumps(prompt_messages, ensure_ascii=False, indent=4)}")
            llm_t0 = time.perf_counter()
            msg = self.chatter.call_llm(messages=prompt_messages, extra_body=self._default_extra_body)
            llm_t1 = time.perf_counter()

            if msg is None:
                logging.error("[Prompt ReAct] LLM returned None (timeout or all retries failed).")
                current_interaction["output"] = {
                    "raw_response": "",
                    "llm_latency_sec": round(llm_t1 - llm_t0, 6),
                    "error": "call_llm_returned_none"
                }
                interaction_log.append(current_interaction)
                return "Error: LLM call failed (None).", interaction_log

            logging.info(f"[Prompt ReAct] LLM response:\n {msg.content}")
            # print(f"[Prompt ReAct] LLM response:\n {msg.content}")
            raw_content = (msg.content or "").strip() 
            current_interaction.setdefault("output", {})
            current_interaction["output"]["raw_response"] = raw_content
            current_interaction["output"]["llm_latency_sec"] = round(llm_t1 - llm_t0, 6)
            
            tool_call_strs = []
            if "<tool_call>" in raw_content:
                parts = raw_content.split("<tool_call>")
                for part in parts[1:]:
                    if "</tool_call>" in part:
                        tool_call_strs.append(part.split("</tool_call>")[0].strip())
                # print(tool_call_strs)

            if not tool_call_strs:
                if "<answer>" in raw_content and "</answer>" in raw_content:
                    try:
                        ans = raw_content.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
                    except:
                        ans = raw_content.split("<answer>", 1)[1].strip()
                    interaction_log.append(current_interaction)
                    return ans, interaction_log
                
                if raw_content:
                    tid = self.memory.record_thought(detailed_reasoning=raw_content, summary_text=None)
                    current_thought_id = str(tid)
                
                interaction_log.append(current_interaction)
                round_num += 1
                continue

            is_commit_only = False
            parsed_calls = []
            for call_str in tool_call_strs:
                try:
                    call_json = json5.loads(call_str)
                    parsed_calls.append(call_json)
                except:
                    continue
            
            if parsed_calls and parsed_calls[0].get("name") == "commit":
                is_commit_only = True

            if is_commit_only:
                commit_args = parsed_calls[0].get("arguments", {})
                current_interaction["tool_calls"].append({
                    "parsed_name": "commit",
                    "parsed_args": commit_args
                })
                # t_sum, o_sum = commit_args.get("thought_summary", ""), commit_args.get("observation_summary", "")
                t_sum, o_sum = commit_args.get("reasoning", ""), commit_args.get("observation_summary", "")
                
                if last_step_thought_id and last_step_obs_id:
                    # self.memory.update_thought_summary(last_step_thought_id, t_sum)
                    # logging.info(f"[Silent Commit] IDs updated: T:{last_step_thought_id}, O:{last_step_obs_id}")

                    self.memory.update_observation_summary(last_step_obs_id, o_sum)
                    logging.info(f"[Silent Commit] IDs updated:  O:{last_step_obs_id}")
                    
                    last_step_thought_id = None
                    last_step_obs_id = None
                    current_thought_id = None
                    current_obs_id = None
                
                current_interaction['notes'].append("Silent commit executed.")
                interaction_log.append(current_interaction)
                round_num += 1
                continue  

            # [debug]
            tid = self.memory.record_thought(detailed_reasoning=raw_content, summary_text=None)
            # tid = self.memory.record_thought(detailed_reasoning=raw_content, summary_text=raw_content)
            current_thought_id = str(tid)
            current_obs_id = None 

            for call_json in parsed_calls:
                name = call_json.get("name")
                args = call_json.get("arguments", {})
                current_interaction['tool_calls'].append({"parsed_name": name, "parsed_args": args})

                if name in {"search", "visit"}:
                    executor = self.tool_executors.get(name)
                    try:
                        observation_content = executor.call(params=args) if executor else f"Tool '{name}' not found"
                        observation_summary_init = None 
                    except Exception as e:
                        observation_content = f"Error executing tool '{name}': {e}"
                        observation_summary_init = f"Error: {e}"
                    
                    obs_xml = f"<tool_response>\n{observation_content}\n</tool_response>"
                    oid_mem = self.memory.record_observation(
                        raw_observation=obs_xml, observation_summary=observation_summary_init, 
                        tool_name=name, tool_args=args
                    )
                    current_obs_id = str(oid_mem)
                    last_step_thought_id = current_thought_id
                    last_step_obs_id = current_obs_id
                # --- 3. EXPAND / FOLD ---
                elif name in {"expand", "fold"}:
                    tid_arg, oid_arg = args.get("thought_id"), args.get("obs_id")
                    reason = args.get("reason", "")
                    obs_parts = []
                    

                    if name == "expand":
                        if tid_arg: currently_expanded_ids["thought"].add(str(tid_arg))
                        if oid_arg: currently_expanded_ids["obs"].add(str(oid_arg))
                        action_past_tense = "expanded"
                        action_view = "further detail"
                    else: # fold
                        if tid_arg: currently_expanded_ids["thought"].discard(str(tid_arg))
                        if oid_arg: currently_expanded_ids["obs"].discard(str(oid_arg))
                        action_past_tense = "folded back"
                        action_view = "summary view"

                    confirm_parts = []
                    if tid_arg: confirm_parts.append(f"Thought ID {tid_arg}")
                    if oid_arg: confirm_parts.append(f"Observation ID {oid_arg}")
                    
                    if confirm_parts:
                        subject = " and ".join(confirm_parts)
                        confirmation_text = f"{subject} has been {action_past_tense} to its {action_view}."
                        obs_parts.append(confirmation_text)

                    observation_content = "\n".join(obs_parts) if obs_parts else f"Invalid {name} call."
                    
                    obs_xml = f"<tool_response>\n{observation_content}\n</tool_response>"
                    current_obs_id = self.memory.record_observation(raw_observation=obs_xml, observation_summary=observation_content)
                    

            interaction_log.append(current_interaction)
            round_num += 1

        msg = "Agent reached maximum rounds without providing an answer."
        logging.warning(msg)

        interaction_log.append({
            "round": round_num,
            "input_messages": copy.deepcopy(prompt_messages) if 'prompt_messages' in locals() else [],
            "output": {},
            "tool_calls": [],
            "notes": [msg]
        })
        return msg, interaction_log


        
    def _rebuild_prompt_messages_with_pending_v1(
        self,
        initial_messages,
        memory,
        expanded_ids,
        pending_t_ids,
        pending_o_ids
    ):
        msgs = list(initial_messages)
        thoughts = dict(memory.get_all_intents())
        obs = dict(memory.get_all_observations())
        
        id_pattern = re.compile(r'^\s*\[(?:Expanded |Folded )?(?:Thought|Observation) ID: \d+\]\s*', re.MULTILINE)
        think_tag_pattern = re.compile(r'</?think>', re.IGNORECASE)

        valid_ids = sorted([int(k) for k in thoughts.keys() if k.isdigit()])
        
        for i in valid_ids:
            sid = str(i)
            t_data = thoughts.get(sid)
            o_data = obs.get(sid)
            if not t_data: continue
                
            raw_original = t_data.get('original', '')
            tool_calls_part = ""
            if "<tool_call>" in raw_original:
                tool_calls_part = raw_original[raw_original.find('<tool_call>'):].strip()
            
            is_mem_op = any(op in tool_calls_part for op in ['\"fold\"', '\"expand\"'])

            is_pending_t = sid in pending_t_ids
            is_expanded_t = sid in expanded_ids["thought"]
            has_summary = bool(t_data.get('summary')) and str(t_data.get('summary', '')).strip() not in ["", "None"]
            
            if is_mem_op:
                id_tag = f"[Folded Thought ID: {sid}]"
            elif is_expanded_t:
                id_tag = f"[Expanded Thought ID: {sid}]"
            elif has_summary and not is_pending_t:
                id_tag = f"[Folded Thought ID: {sid}]"
            else:
                id_tag = f"[Thought ID: {sid}]"

            should_show_original = is_mem_op or is_pending_t or is_expanded_t or not has_summary
            
            if should_show_original:
                thought_part = raw_original.split('<tool_call>')[0] if tool_calls_part else raw_original
                clean_thought = id_pattern.sub('', thought_part).strip()
                clean_thought = think_tag_pattern.sub('', clean_thought).strip()
            else:
                clean_thought = str(t_data.get('summary', '')).strip()

            final_content = f"<think>\n{id_tag}\n{clean_thought}\n</think>"
            if tool_calls_part:
                final_content += "\n\n" + tool_calls_part
            
            msgs.append({"role": "assistant", "content": final_content})

            if o_data:
                is_pending_o = sid in pending_o_ids
                is_expanded_o = sid in expanded_ids["obs"]
                has_o_summary = bool(o_data.get('summary')) and str(o_data.get('summary', '')).strip() not in ["", "None"]
                
                if is_mem_op:
                    id_tag_o = f"[Folded Observation ID: {sid}]"
                elif is_expanded_o:
                    id_tag_o = f"[Expanded Observation ID: {sid}]"
                elif has_o_summary and not is_pending_o:
                    id_tag_o = f"[Folded Observation ID: {sid}]"
                else:
                    id_tag_o = f"[Observation ID: {sid}]"

                should_show_o_original = is_mem_op or is_pending_o or is_expanded_o or not has_o_summary
                
                if should_show_o_original:
                    raw_obs_xml = o_data.get('original', '')
                    inner_match = re.search(r'<tool_response>(.*?)</tool_response>', raw_obs_xml, re.DOTALL)
                    inner_content = inner_match.group(1).strip() if inner_match else raw_obs_xml.strip()
                    inner_content = id_pattern.sub('', inner_content).strip()
                    content_o = f"<tool_response>\n{id_tag_o}\n{inner_content}\n</tool_response>"
                else:
                    summ_o = str(o_data.get('summary', '')).strip()
                    content_o = f"<tool_response>\n{id_tag_o}\n{summ_o}\n</tool_response>"
                
                msgs.append({"role": "user", "content": content_o})
                    
        return msgs




    def _format_history_to_string(self, messages: list) -> str:
        """
        Question: ...
        Assistant: content + <tool_call>...</tool_call>
        <tool_response>...</tool_response>
        """
        formatted_parts = []
        
        try:
            original_q = next(m['content'] for m in messages if m['role'] == 'user')
            formatted_parts.append(f"Question: {original_q}")
        except StopIteration:
            pass

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            
            if role == "assistant":
                reasoning = msg.get("reasoning_content") or ""
                text_output = f"{reasoning}\n{content}".strip()
                
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    tool_calls_str = json.dumps(tool_calls, ensure_ascii=False)
                    formatted_parts.append(f"{text_output}\n<tool_call>{tool_calls_str}</tool_call>")
                else:
                    if text_output:
                        formatted_parts.append(f"{text_output}")

            elif role == "tool":
                formatted_parts.append(f"<tool_response>\n{content}\n</tool_response>")
                
            elif role == "user" and content != original_q:
                formatted_parts.append(f"{content}")

        return "\n\n".join(formatted_parts)
        
    def _run_turn_dual_inference_fn_call(self, history: list) -> tuple:
        """
        （Main + Logger）DeepResearch 

        - Logger ：， JSON（ function calling），
         last_raw_thought / last_raw_observation  self.memory。
        - Main ： function calling/tools  search / visit / expand / fold 。
        - ：
            - token  MAX_REASON_TOKENS ，， <answer>your answer</answer> ；
            -  MAX_ROUNDS / 。
        """

        interaction_log = []

        try:
            original_user_question = next(item['content'] for item in history if item['role'] == 'user')
        except StopIteration:
            return "Error: No user question found in initial history.", []

        round_num = 1
        MAX_ROUNDS = int(os.getenv('MAX_ROUNDS', 100))

        last_raw_thought: Optional[str] = None
        last_raw_observation: Optional[str] = None
        last_thought_id: Optional[str] = None
        last_obs_id: Optional[str] = None
        last_tool_name: Optional[str] = None
        last_tool_args: Optional[Dict] = None
        last_tool_type: Optional[str] = None
        last_main_messages: List[Dict] = []
        next_step_user_msg: str = ""
        currently_expanded_ids: Dict[str, set] = {'thought': set(), 'obs': set()}

        no_tool_retry_count = 0
        MAX_NO_TOOL_RETRIES = 5
        retry_feedback_message: Optional[Dict] = None

        allowed_tool_names = {"search", "visit", "expand", "fold"}
        active_tools = [
            tool for tool in self.tools
            if tool['function']['name'] in allowed_tool_names
        ]
        main_agent_tools = active_tools
        logging.info(f"[DeepResearch] Main Agent Tools: {[tool['function']['name'] for tool in main_agent_tools]}")

        start_time = time.time()
        max_tokens = int(os.getenv("MAX_REASON_TOKENS", 108 * 1024))


        while round_num <= MAX_ROUNDS:
            if time.time() - start_time > 150 * 60:
                timeout_msg = "No answer found after 2h30mins (dual inference)."
                logging.warning(timeout_msg)
                interaction_log.append({
                    "round": round_num,
                    "input_messages": [],
                    "output": {},
                    "tool_calls": [],
                    "notes": [timeout_msg]
                })
                return timeout_msg, interaction_log

            round_header = f"--- [DEEP RESEARCH] Round {round_num}/{MAX_ROUNDS} ---"
            print(f"\n{'=' * len(round_header)}\n{round_header}\n{'=' * len(round_header)}")
            logging.info(round_header)

            current_interaction = {
                "round": round_num,
                "input_messages": [],
                "output": {},
                "tool_calls": [],
                "notes": []
            }

            # ======================================================
            # ======================================================
            if last_raw_thought is not None and last_raw_observation is not None:
                print("[Coordinator] Pending record detected. Calling Logger Model...")
                current_interaction['notes'].append("Phase: Logging")
                if last_tool_type == "search":
                    logger_system_prompt = self.search_logger_system_prompt
                elif last_tool_type == "visit":
                    logger_system_prompt = self.visit_logger_system_prompt
                else:
                    error_note = f"Unknown last_tool_type: {last_tool_type}. Skipping logging for this step."
                    logging.error(error_note)
                    current_interaction["notes"].append(error_note)
                    interaction_log.append(current_interaction)
                    last_raw_thought, last_raw_observation = None, None
                    last_thought_id, last_obs_id = None, None
                    last_tool_name, last_tool_args, last_tool_type = None, None, None
                    round_num += 1
                    continue

                
                logger_user_content = (
                    "You are processing a search result in the context of a larger research goal.\n"
                    f"## Overall Research Goal:\n{original_user_question}\n\n" 
                    "----- Current Temporary Action Record -----\n"
                    f"## Tool Name: {last_tool_name}\n"
                    f"## Tool Arguments: {json.dumps(last_tool_args, ensure_ascii=False)}\n"
                    f"## Raw Thought:\n{last_raw_thought}\n"
                    f"## Raw Observation:\n{last_raw_observation}\n"
                    "----- End of Record -----\n\n"
                    "Please synthesize the information from the Raw Observation that is relevant to the Overall Research Goal."
                )

                if os.getenv("LOGGER_WITH_CONTEXT", "0") == "1"  and last_main_messages:
                    prompt_messages = [
                        {"role": "system", "content": logger_system_prompt}
                    ] + last_main_messages[1:] + [
                        {"role": "user", "content": logger_user_content}
                    ]
                    logging.info("[Coordinator] Logger calling with FULL CONTEXT.")
                else:
                    prompt_messages = [
                        {"role": "system", "content": logger_system_prompt},
                        {"role": "user", "content": logger_user_content},
                    ]

                logger_json_retry_count = 0
                LOGGER_MAX_JSON_RETRIES = 3

                while logger_json_retry_count <= LOGGER_MAX_JSON_RETRIES:
                    current_interaction["input_messages"] = copy.deepcopy(prompt_messages)
                    # logging.info(f"[Logger] Prompt messages: {json.dumps(prompt_messages, ensure_ascii=False, indent=2)}")
                    msg = self.summarizer_chatter.call_llm(
                        messages=prompt_messages,
                    )
                    if msg is None:
                        error_note = "[Logger] summarizer_chatter.call_llm returned None after retries. Skipping logging for this step."
                        logging.error(error_note)
                        current_interaction["notes"].append(error_note)
                        interaction_log.append(current_interaction)
                        last_raw_thought, last_raw_observation = None, None
                        last_thought_id, last_obs_id = None, None
                        last_tool_name, last_tool_args, last_tool_type = None, None, None
                        break

                    msg_dict = msg.model_dump(exclude_none=True)
                    current_interaction["output"] = {"raw_response": msg_dict}
                    # logging.info(f"[Logger] LLM output: {msg.content}")
                    try:
                        parsed_args = self._parse_logger_text_output(msg.content or "")
                        if not parsed_args:
                            raise ValueError("Failed to parse logger text output.")

                        reasoning = parsed_args.get("reasoning")
                        obs_summary = parsed_args.get("observation_summary")
                        if not (reasoning and obs_summary):
                            raise ValueError("Missing 'reasoning' or 'observation_summary' in logger output.")

                        print("[Logger] Successfully validated arguments from logger content.")

                        if last_thought_id is None and last_obs_id is None:
                            note = "[Logger] Got valid summaries but last_thought_id/last_obs_id are None. Skipping memory update for this step."
                            logging.warning(note)
                            current_interaction["notes"].append(note)

                            last_raw_thought, last_raw_observation = None, None
                            last_thought_id, last_obs_id = None, None
                            last_tool_name, last_tool_args, last_tool_type = None, None, None

                            # logging.info(f"[Logger] : {json.dumps(current_interaction, ensure_ascii=False, indent=2)}")
                            break

                        if last_thought_id is None or last_obs_id is None:
                            note = f"[Logger] Partial IDs: thought_id={last_thought_id}, obs_id={last_obs_id}. Updating available ones."
                            logging.warning(note)
                            current_interaction["notes"].append(note)

                        logging.debug(
                            f"[Logger] Updating summaries Thought ID: {last_thought_id}, Obs ID: {last_obs_id}"
                        )
                        # if last_thought_id is not None:
                        #     self.memory.update_thought_summary(last_thought_id, thought_summary)

                        if last_obs_id is not None:
                            self.memory.update_observation_summary(last_obs_id, obs_summary)

                        tool_call_record = {
                            "tool_name": "log_step (from content)",
                            "arguments": parsed_args,
                            "status": "success",
                            "thought_id": last_thought_id,
                            "obs_id": last_obs_id
                        }
                        current_interaction["tool_calls"].append(tool_call_record)
                        print(f"[Logger] Successfully logged step. Thought ID: {last_thought_id}, Obs ID: {last_obs_id}")

                        last_raw_thought, last_raw_observation = None, None
                        last_thought_id, last_obs_id = None, None
                        last_tool_name, last_tool_args, last_tool_type = None, None, None

                        # logging.info(f"[Logger] : {json.dumps(current_interaction, ensure_ascii=False, indent=2)}")
                        break

                    except Exception as e:
                        logger_json_retry_count += 1
                        error_details = f"Error: {e}. Raw output: '{msg.content}'"

                        if logger_json_retry_count > LOGGER_MAX_JSON_RETRIES:
                            error_note = (
                                f"CRITICAL FAILURE: Logger failed after {LOGGER_MAX_JSON_RETRIES} retries. "
                                f"{error_details}"
                            )
                            print(f"[Coordinator] {error_note}")
                            logging.error(error_note)
                            current_interaction["notes"].append(error_note)
                            interaction_log.append(current_interaction)

                            last_raw_thought, last_raw_observation = None, None
                            last_thought_id, last_obs_id = None, None
                            last_tool_name, last_tool_args, last_tool_type = None, None, None

                            logging.info("[Coordinator] Skipping logging for this step and returning to Main phase.")
                            break

                        note = f"Logger invalid output. Retry {logger_json_retry_count}/{LOGGER_MAX_JSON_RETRIES}."
                        print(f"[Coordinator] {note} Details: {error_details}")
                        logging.warning(f"{note} {error_details}")
                        logging.warning(f"[Logger] Input messages: {json.dumps(prompt_messages, ensure_ascii=False, indent=2)}")

                next_step_user_msg = ""

            # ======================================================
            # ======================================================
            else:
                print("[Coordinator] Clean state. Calling Main Model for next action...")
                current_interaction['notes'].append("Phase: Thinking/Action")

                main_agent_initial_messages = [
                    {"role": "system", "content": self.main_system_prompt + today_date()},
                    {"role": "user", "content": original_user_question}
                ]

                prompt_messages = self._rebuild_prompt_messages_fn_call(
                    main_agent_initial_messages,
                    self.memory,
                    currently_expanded_ids,
                    next_step_user_msg,
                    mode="merged"
                )

                next_step_user_msg = ""
                if retry_feedback_message:
                    prompt_messages.append(retry_feedback_message)
                    retry_feedback_message = None

                current_interaction["input_messages"] = copy.deepcopy(prompt_messages)
                last_main_messages = copy.deepcopy(prompt_messages)
                # print(f"[Main Agent] Prompt messages: {json.dumps(prompt_messages, ensure_ascii=False, indent=2)}")

                # print(main_agent_tools)
                msg = self.chatter.call_llm(
                    messages=prompt_messages,
                    tools=main_agent_tools,
                    extra_body=self._default_extra_body
                )
                if msg is None:
                    error_msg = "[DeepResearch] call_llm returned None after retries in Main phase. Aborting this sample."
                    logging.error(error_msg)
                    current_interaction["notes"].append(error_msg)
                    interaction_log.append(current_interaction)
                    return error_msg, interaction_log

                msg_dict = msg.model_dump(exclude_none=True)
                thinking_content = (
                    msg_dict.get("reasoning_content")
                    or msg_dict.get("content")
                    or "No thinking content provided."
                )


                clean_summary = self._strip_thought_tags(thinking_content)
                if clean_summary != thinking_content:
                    logging.info(f"[DeepResearch] Main Model Thinking Content (raw): {thinking_content}")
                    logging.info(f"[DeepResearch] Main Model Thinking Content (cleaned): {clean_summary}")

                current_interaction["output"] = {
                    "raw_response": msg_dict
                }

                tool_calls = msg_dict.get("tool_calls")

                if not tool_calls:
                    content = msg.content or ""
                    if "<answer>" in content:
                        start_index = content.find("<answer>")
                        answer_text = content[start_index + len("<answer>"):]
                        if "</answer>" in answer_text:
                            end_index = answer_text.rfind("</answer>")
                            final_answer_text = answer_text[:end_index].strip()
                        else:
                            final_answer_text = answer_text.strip()

                        note = f"Final answer detected in <answer> tag: '{final_answer_text}'"
                        print(f"[Coordinator] {note}")
                        logging.info(note)
                        current_interaction['notes'].append(note)
                        interaction_log.append(current_interaction)
                        return final_answer_text, interaction_log

                    no_tool_retry_count += 1
                    if no_tool_retry_count > MAX_NO_TOOL_RETRIES:
                        error_msg = (
                            f"Agent failed to respond with a tool call after "
                            f"{MAX_NO_TOOL_RETRIES} retries. Halting."
                        )
                        logging.error(error_msg)
                        current_interaction['notes'].append(error_msg)
                        interaction_log.append(current_interaction)
                        return msg.content or error_msg, interaction_log

                    note = f"No tool call returned. Retry attempt {no_tool_retry_count}/{MAX_NO_TOOL_RETRIES}."
                    available_tool_names = [tool['function']['name'] for tool in main_agent_tools]
                    formatted_tools = ", ".join([f"`{name}`" for name in available_tool_names])
                    feedback_content = (
                        "You must call a function to proceed or choose to give your final answer using the format: <answer>your answer</answer>."
                        f"\nAvailable tools: {formatted_tools}."
                    )

                    current_interaction['notes'].append(note)
                    interaction_log.append(current_interaction)
                    logging.warning(note)
                    logging.info(f"[DeepResearch] | no tool calls : {msg_dict}")
                    print(f"[Coordinator] {note}")
                    # print(f"[DeepResearch] | no tool calls : {msg_dict}")

                    retry_feedback_message = {"role": "user", "content": feedback_content}
                    round_num += 1
                    continue

                no_tool_retry_count = 0

                thought_recorded = False

                for call in tool_calls:
                    name, call_id, args = self._parse_tool_call(call)
                    if args is None:
                        continue

                    if not thought_recorded:
                        last_thought_id = self.memory.record_thought(
                            detailed_reasoning=clean_summary,
                            summary_text=clean_summary,
                            tool_calls=tool_calls
                        )
                        thought_recorded = True

                    tool_call_record = {
                        "tool_name": name,
                        "arguments": args,
                        "call_id": call_id
                    }
                    print(f"[Main Agent] Executing tool: {name}")

                    if name in {"search", "visit"}:
                        executor = self.tool_executors.get(name)
                        try:
                            observation = executor.call(params=args)
                            tool_call_record["status"] = "success"

                            last_obs_id = self.memory.record_observation(
                                observation_summary=str(observation),
                                raw_observation=observation,
                                tool_name=name,
                                tool_args=args,
                                tool_call_id=call_id,
                            )

                            last_raw_thought = thinking_content
                            last_raw_observation = observation
                            last_tool_name = name
                            last_tool_args = args
                            last_tool_type = name  

                        except Exception as e:
                            error_text = f"Error executing tool '{name}': {e}"
                            tool_call_record["status"] = "error"

                            obs_id = self.memory.record_observation(
                                observation_summary=error_text,
                                raw_observation=error_text,
                                tool_name=name,
                                tool_args=args,
                                tool_call_id=call_id,
                            )
                            last_obs_id = obs_id
                            last_raw_thought = thinking_content
                            last_raw_observation = error_text
                            last_tool_name = name
                            last_tool_args = args
                            last_tool_type = name

                    elif name == "expand":
                        tid, oid = args.get("thought_id"), args.get("obs_id")
                        reason = args.get("reason", "")

                        if tid:
                            currently_expanded_ids['thought'].add(str(tid))
                        if oid:
                            currently_expanded_ids['obs'].add(str(oid))
                        tool_call_record["status"] = "success"

                        # if reason:
                        #     reason = reason.strip()
                        #         reason += '.'
                        #     if tid and oid:
                        #         expand_msg = (
                        #             f"{reason} Thought ID {tid} and Observation ID {oid} "
                        #             "have been expanded for further detail."
                        #         )
                        #     elif tid:
                        #         expand_msg = (
                        #             f"{reason} Thought ID {tid} has been expanded for further detail."
                        #         )
                        #     elif oid:
                        #         expand_msg = (
                        #             f"{reason} Observation ID {oid} has been expanded for further detail."
                        #         )
                        # else:
                        expand_msg = ""
                        if tid and oid:
                            expand_msg = (
                                f"Thought ID {tid} and Observation ID {oid} "
                                "have been expanded for further detail."
                            )
                        elif tid:
                            expand_msg = f"Thought ID {tid} has been expanded for further detail."
                        elif oid:
                            expand_msg = f"Observation ID {oid} has been expanded for further detail."

                        if expand_msg:
                            next_step_user_msg = expand_msg

                            self.memory.record_observation(
                                observation_summary=expand_msg,
                                raw_observation=expand_msg,
                                tool_name="expand",
                                tool_args=args,
                                tool_call_id=call_id,
                            )

                    elif name == "fold":
                        tid, oid = args.get("thought_id"), args.get("obs_id")

                        if tid:
                            currently_expanded_ids['thought'].discard(str(tid))
                        if oid:
                            currently_expanded_ids['obs'].discard(str(oid))
                        tool_call_record["status"] = "success"

                        fold_msg = ""
                        if tid and oid:
                            fold_msg = (
                                f"Thought ID {tid} and Observation ID {oid} "
                                "have been folded back to summary view."
                            )
                        elif tid:
                            fold_msg = f"Thought ID {tid} has been folded back to summary view."
                        elif oid:
                            fold_msg = f"Observation ID {oid} has been folded back to summary view."

                        if fold_msg:
                            next_step_user_msg = fold_msg

                            self.memory.record_observation(
                                observation_summary=fold_msg,
                                raw_observation=fold_msg,
                                tool_name="fold",
                                tool_args=args,
                                tool_call_id=call_id,
                            )

                    else:
                        tool_call_record["status"] = "error"
                        logging.warning(f"[Main Agent] Unsupported tool called: {name}")

                    current_interaction["tool_calls"].append(tool_call_record)

                token_count = self.count_tokens(prompt_messages)
                logging.info(f"[DualInference] round {round_num}, token count: {token_count}")
                if token_count > max_tokens:
                    note = f"Token quantity exceeds the limit: {token_count} > {max_tokens}"
                    logging.warning(note)
                    current_interaction["notes"].append(note)

                    final_prompt = (
                        "You have now reached the maximum context length you can handle. "
                        "You should stop making tool calls and, based on all the information above, "
                        "think again and provide what you consider the most likely answer. "
                        "You should directly output your answer in the format: <answer>your answer</answer>."
                    )
                    prompt_messages.append({"role": "user", "content": final_prompt})

                    msg_final = self.chatter.call_llm(
                        messages=prompt_messages,
                        extra_body=self._default_extra_body
                    )
                    if msg_final is None:
                        note = "[DualInference] Final call after token limit returned None. Returning error answer."
                        logging.error(note)
                        current_interaction["notes"].append(note)
                        interaction_log.append(current_interaction)
                        return "Token limit reached but no answer provided (LLM failed).", interaction_log

                    final_dict = msg_final.model_dump(exclude_none=True)
                    final_content = (msg_final.content or "").strip()

                    final_interaction = {
                        "round": round_num + 1,
                        "input_messages": copy.deepcopy(prompt_messages),
                        "output": {"raw_response": final_dict},
                        "tool_calls": [],
                        "notes": ["Forced final answer due to token limit"]
                    }
                    interaction_log.append(current_interaction)
                    interaction_log.append(final_interaction)

                    prompt_messages.append({"role": "assistant", "content": final_content})

                    if "<answer>" in final_content:
                        try:
                            ans_text = final_content.split("<answer>", 1)[1]
                            if "</answer>" in ans_text:
                                final_answer_text = ans_text.split("</answer>", 1)[0].strip()
                            else:
                                final_answer_text = ans_text.strip()
                        except Exception:
                            final_answer_text = final_content
                        logging.info(f"[Coordinator] Final answer (token limit via content): {final_answer_text}")
                        return final_answer_text, interaction_log

                    logging.warning("[Coordinator] Token limit reached but no valid final answer provided.")
                    return final_content or "Token limit reached but no answer provided.", interaction_log

            interaction_log.append(current_interaction)
            logging.info(f"[DeepResearch] Completed round {round_num}.")
            logging.info(f"[DeepResearch] : {json.dumps(current_interaction, ensure_ascii=False, indent=2)}")
            print(f"[DeepResearch] Completed round {round_num}.")
            # print(f"[DeepResearch] : {json.dumps(current_interaction, ensure_ascii=False, indent=2)}")

            round_num += 1

        return "Agent reached maximum rounds without providing an answer.", interaction_log

    def _parse_logger_text_output(self, output_text: str) -> dict:
        """
         Logger 。 </observation_summary> 
         ')' 。
        """
        output_text = output_text.strip()
        
        try:
            start_tag_reason = "<reasoning>"
            end_tag_reason = "</reasoning>"
            
            s_r = output_text.find(start_tag_reason)
            e_r = output_text.find(end_tag_reason)
            
            if s_r == -1 or e_r == -1 or e_r <= s_r:
                raise ValueError("Missing or invalid <reasoning> tags.")
            
            reasoning = output_text[s_r + len(start_tag_reason) : e_r].strip()

            start_tag_obs = "<observation_summary>"
            end_tag_obs = "</observation_summary>"
            
            s_o = output_text.find(start_tag_obs)
            if s_o == -1:
                raise ValueError("Could not find start tag <observation_summary>.")

            e_o = output_text.find(end_tag_obs, s_o)
            
            if e_o != -1:
                observation_summary = output_text[s_o + len(start_tag_obs) : e_o].strip()
            else:
                remaining_content = output_text[s_o + len(start_tag_obs):].strip()
                
                if remaining_content.endswith(')'):
                    observation_summary = remaining_content
                    logging.info("[Logger Parser] Missing </observation_summary> but found ')' at end. Applying soft-parse.")
                else:
                    raise ValueError("Could not find end tag </observation_summary> and content does not end with ')'.")

            if not reasoning or not observation_summary:
                raise ValueError("One of the required fields is empty.")

            return {
                "reasoning": reasoning,
                "observation_summary": observation_summary
            }
        
        except Exception as e:
            raise ValueError(f"Failed to parse logger output: {e}")
    

    def _rebuild_prompt_messages_fn_call(
        self,
        initial_messages: List[Dict],
        memory,
        currently_expanded_ids: Dict[str, set],
        next_step_user_msg: str,
        mode: str = "merged"
    ) -> List[Dict]:

        if mode != "merged":
            return list(initial_messages)

        prompt_messages = list(initial_messages)

        all_thoughts = dict(memory.get_all_intents())
        all_observations = dict(memory.get_all_observations())

        history_items = []

        max_id = 0
        if all_thoughts and all_observations:
            valid_thought_ids = {int(k) for k in all_thoughts.keys() if k.isdigit()}
            valid_obs_ids = {int(k) for k in all_observations.keys() if k.isdigit()}
            if valid_thought_ids and valid_obs_ids:
                max_id = min(max(valid_thought_ids), max(valid_obs_ids))

        for i in range(1, max_id + 1):
            cycle_id = str(i)

            thought_record = all_thoughts.get(cycle_id) or {}
            obs_record = all_observations.get(cycle_id) or {}

            raw_thought_summary = thought_record.get(
                "summary",
                thought_record.get("original", "No summary.")
            ) or ""

            if cycle_id in currently_expanded_ids.get('thought', set()):
                thought_text = (
                    f"[Expanded Thought ID: {cycle_id}]\n"
                    f"{thought_record.get('original', 'Details not found.')}"
                )
            else:
                stripped = raw_thought_summary.lstrip()
                if stripped.startswith("[Thought ID:") or stripped.startswith("[Expanded Thought ID:"):
                    thought_text = raw_thought_summary
                else:
                    thought_text = f"[Thought ID: {cycle_id}] {raw_thought_summary}"

            if cycle_id in currently_expanded_ids.get('obs', set()):
                obs_text = (
                    f"[Expanded Observation ID: {cycle_id}]\n"
                    f"{obs_record.get('original', 'Raw data not found.')}"
                )
            else:
                obs_text = (
                    f"[Observation ID: {cycle_id}] "
                    f"{obs_record.get('summary', obs_record.get('original', 'No summary.'))}"
                )

            history_items.append({
                "id": cycle_id,
                "thought_text": thought_text,
                "obs_text": obs_text,
                "thought_tool_calls": thought_record.get("tool_calls"),
                "tool_call_id": obs_record.get("tool_call_id"),
            })

        for item in history_items:
            if item["thought_text"]:
                assistant_msg = {
                    "role": "assistant",
                    "content": item["thought_text"],
                }
                if item["thought_tool_calls"]:
                    assistant_msg["tool_calls"] = item["thought_tool_calls"]
                prompt_messages.append(assistant_msg)

            if item["obs_text"]:
                tool_msg = {
                    "role": "tool",
                    "content": item["obs_text"],
                }
                if item["tool_call_id"]:
                    tool_msg["tool_call_id"] = item["tool_call_id"]
                prompt_messages.append(tool_msg)

        return prompt_messages

    def _parse_tool_call(self, call: Dict) -> (str, str, Dict[str, Any]):
        name = call['function']['name']
        call_id = call['id']
        try:
            args = json.loads(call['function']['arguments'])
            return name, call_id, args
        except json.JSONDecodeError:
            logging.warning(f"Failed to decode arguments for tool {name}: {call['function']['arguments']}")
            return name, call_id, None

    def count_tokens(self, messages, model: str = "gpt-4o") -> int:
        """
         OpenAI  messages（[{role, content, ...}, ...]） token 。

        - messages ， role="tool" 。
        -  HF tokenizer（./tokenizer/）， tiktoken。
        -  messages  role + content ，。
        """
        try:
            from transformers import AutoTokenizer
            _default_tok_path = os.path.join(
                os.path.dirname(__file__), "tokenizer", "Qwen3-30B-A3B-Thinking-2507"
            )
            model_path = os.environ.get("TOKENIZER_PATH", _default_tok_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
            )
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(full_prompt, return_tensors="pt")
            return tokens["input_ids"].shape[1]
        except Exception:
            pass

        import tiktoken
        logging.error("[count_tokens] HF tokenizer failed. Falling back to tiktoken.")
        enc = tiktoken.encoding_for_model(model)

        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    str(b.get("text") or b.get("content") or "") if isinstance(b, dict) else str(b)
                    for b in content
                ]
                content_str = "\n".join(texts)
            else:
                content_str = str(content)
            parts.append(f"{role}: {content_str}")

        full_prompt = "\n\n".join(parts)
        return len(enc.encode(full_prompt))

    def _strip_thought_tags(self, text: str) -> str:
        """
         Thought/Observation ，：
        - "[Thought ID: 6] ..."
        - "[Expanded Thought ID: 6] ..."
        - "[Observation ID: 3] ..."
        - "[Expanded Observation ID: 3] ..."

        """
        if not isinstance(text, str):
            return text

        s = text.lstrip()

        s = re.sub(r'^\[(?:Expanded )?Thought ID:\s*\d+\]\s*', '', s)
        s = re.sub(r'^\[(?:Expanded )?Observation ID:\s*\d+\]\s*', '', s)

        return s
    
    def evaluate_responses(self, max_workers: int = 10, skip_if_exists: bool = True):
        if skip_if_exists and getattr(self, "final_report_file", None) and os.path.exists(self.final_report_file):
            logging.info(f"Final report already exists at {self.final_report_file}. Skip evaluation.")
            return

        if not os.path.exists(self.raw_output_file):
            logging.error(f"Raw output file not found at {self.raw_output_file}. Nothing to evaluate.")
            return

        try:
            with open(self.raw_output_file, 'r', encoding='utf-8') as f:
                examples = [json.loads(line) for line in f.readlines()]
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Failed to read or parse {self.raw_output_file}: {e}")
            return

        if not examples:
            logging.warning(f"Raw output file {self.raw_output_file} is empty.")
            return

        def _extract_answer_tag(predicted_answer: str) -> str:
            if not predicted_answer:
                return ""
            start_tag = "<answer>"
            start_index = predicted_answer.find(start_tag)
            if start_index == -1:
                return predicted_answer
            answer_text = predicted_answer[start_index + len(start_tag):]
            end_tag = "</answer>"
            end_index = answer_text.rfind(end_tag)
            return (answer_text[:end_index] if end_index != -1 else answer_text).strip()

        def _build_prompt(question: str, correct_answer: str, predicted_answer: str) -> str:
            if self.task_name == "DeepResearch":
                return JUDGE_PROMPT_BROWSECOMP_OFFICIAL.format(
                    question=question, correct_answer=correct_answer, response=predicted_answer
                )
            elif self.task_name in ["BC_en", "BC_zh"]:
                return JUDGE_PROMPT_BROWSECOMP_OFFICIAL.format(
                    question=question, correct_answer=correct_answer, response=predicted_answer
                )
            elif self.task_name.startswith("xbench"):
                return JUDGE_PROMPT_XBENCH.format(
                    question=question, correct_answer=correct_answer, response=predicted_answer
                )
            else:
                return JUDGE_PROMPT_BROWSECOMP_OFFICIAL.format(
                    question=question, correct_answer=correct_answer, response=predicted_answer
                )

        def _eval_one(example: dict) -> dict:
            """
             detail dict（ evaluation_result / evaluation_reasoning）
            evaluation_result ：Correct / Incorrect / Skipped / Error (...)
            """
            detail = example.copy()

            question = example.get("question")
            correct_answer = example.get("answer")
            predicted_answer = _extract_answer_tag(example.get("model_answer", "") or "")

            if not all([question, correct_answer]):
                detail["evaluation_result"] = "Skipped"
                detail["evaluation_reasoning"] = "Missing question or correct_answer."
                return detail

            if not predicted_answer.strip():
                detail["evaluation_result"] = "Error (Empty Answer)"
                detail["evaluation_reasoning"] = "The model failed to produce a final answer."
                return detail

            prompt = _build_prompt(question, correct_answer, predicted_answer)

            for attempt in range(3):
                try:
                    if self.task_name.startswith("xbench"):
                        response = litellm.completion(
                            model=os.getenv("EVALUATE_MODEL_NAME"),
                            api_base=os.getenv("EVALUATE_API_BASE"),
                            api_key=os.getenv("EVALUATE_API_KEY"),
                            messages=[{"role": "user", "content": prompt}],
                            api_version="2025-07-01-preview",
                            num_retries=5,
                            response_format=extracted_answer_format_for_xbench
                        )
                        raw_content = response.choices[0].message.content
                        raw_judge = json.loads(raw_content)
                        judgement_val = raw_judge.get("", "")

                        if judgement_val == "":
                            detail["evaluation_result"] = "Correct"
                        elif judgement_val == "":
                            detail["evaluation_result"] = "Incorrect"
                        else:
                            raise ValueError(f"Invalid '' value: {judgement_val}")

                        detail["evaluation_reasoning"] = raw_judge
                        return detail
                    else:
                        response = litellm.completion(
                            model=os.getenv("EVALUATE_MODEL_NAME"),
                            api_base=os.getenv("EVALUATE_API_BASE"),
                            api_key=os.getenv("EVALUATE_API_KEY"),
                            messages=[{"role": "user", "content": prompt}],
                            api_version="2025-07-01-preview",
                            num_retries=5,
                            response_format=extracted_answer_format_for_confidence
                        )
                        raw_content = response.choices[0].message.content
                        raw_judge = json.loads(raw_content)
                        judgement_str = str(raw_judge.get("correct", "")).lower()

                        if judgement_str == "yes":
                            detail["evaluation_result"] = "Correct"
                        elif judgement_str == "no":
                            detail["evaluation_result"] = "Incorrect"
                        else:
                            raise ValueError(f"Invalid 'correct' value: {judgement_str}")

                        detail["evaluation_reasoning"] = raw_judge
                        return detail

                except Exception as e:
                    logging.warning(f"Judge attempt {attempt + 1} failed for example {example.get('id','N/A')}: {e}")
                    if attempt < 2:
                        time.sleep(3)

            detail["evaluation_result"] = "Error (Judge API Failed)"
            detail["evaluation_reasoning"] = "Failed after 3 attempts."
            return detail

        evaluation_details = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_eval_one, exm) for exm in examples]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Evaluating with Judge (parallel)"):
                evaluation_details.append(fut.result())

        num_correct = sum(1 for d in evaluation_details if d.get("evaluation_result") == "Correct")
        num_incorrect = sum(1 for d in evaluation_details if d.get("evaluation_result") == "Incorrect")
        num_errors = sum(1 for d in evaluation_details if str(d.get("evaluation_result", "")).startswith("Error"))
        num_not_attempted = 0
        total_evaluated = sum(1 for d in evaluation_details if d.get("evaluation_result") != "Skipped")

        accuracy = num_correct / total_evaluated if total_evaluated > 0 else 0

        final_report = {
            "task_name": "DeepResearch" if "DeepResearch" in self.__class__.__name__ else self.output_base_name,
            "num_total_samples": len(examples),
            "num_evaluated": total_evaluated,
            "summary": {
                "Correct": num_correct,
                "Incorrect": num_incorrect,
                "Not Attempted": num_not_attempted,
                "Errors": num_errors
            },
            "accuracy": accuracy,
            "details": evaluation_details
        }

        with open(self.final_report_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)

        logging.info(f"Evaluation complete. Accuracy: {accuracy:.2%}. Report saved to {self.final_report_file}")
        logging.info(f"Summary: Correct={num_correct}, Incorrect={num_incorrect}, Not Attempted={num_not_attempted}, Errors={num_errors}")
