import os
import re
import json
import random
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import transformers
import concurrent.futures
import time

from datasets import load_dataset, get_dataset_config_names
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .prompts import *
from .infer_utils import parse_model_output

DATA_DIR = "data/eval"

class ReasoningStep:
    def __init__(self, detailed_reasoning: str, step_id: int, summary: str):
        self.step_id = step_id
        self.detailed_reasoning = detailed_reasoning
        self.summary = summary
        self.expanded: bool = False

    def mark_expanded(self) -> None:
        self.expanded = True
    
    def mark_folded(self) -> None:
        self.expanded = False

    def is_expanded(self) -> bool:
        return self.expanded

class ScratchPad:
    """Store intermediate reasoning steps."""
    def __init__(self):
        self.steps: List[ReasoningStep] = []

    def add_step(self, detailed_reasoning: str, summary: str) -> None:
        if not detailed_reasoning or not summary:
            return
        id = len(self.steps) + 1
        step = ReasoningStep(detailed_reasoning, id, summary)
        self.steps.append(step)

    def expand_step(self, step_id: int) -> None:
        if 0 < step_id <= len(self.steps):
            self.steps[step_id - 1].mark_expanded()
    
    def fold_step(self, step_id: int) -> None:
        if 0 < step_id <= len(self.steps):
            self.steps[step_id - 1].mark_folded()

    def build_history_text_for_prompt(self) -> str:
        history_lines = []
        for step in self.steps:
            if step.is_expanded():
                history_lines.append(f"[Expanded Step {step.step_id}] {step.detailed_reasoning}\n")
            else:
                history_lines.append(f"[Step {step.step_id}] {step.summary}\n")
        return "".join(history_lines).strip()
    
    def clear(self) -> None:
        self.steps = []

# ==========================================
# Base : BaseInteractiveBenchmark
# ==========================================
class BaseInteractiveBenchmark:
    def __init__(self, **kwargs):
        # General settings
        self.dataset_name = kwargs.get("dataset_name", "gpqa")
        self.subset_name = kwargs.get("subset_name", None)
        self.debug = kwargs.get("debug", False)
        self.debug_num = kwargs.get("debug_num", 2)
        self.start_idx = kwargs.get("start_idx", 0)
        self.end_idx = kwargs.get("end_idx", None)
        self.max_tokens = kwargs.get("max_tokens", 10240)
        self.turn_max_tokens = kwargs.get("turn_max_tokens", 1024)
        self.max_tokens_setting = kwargs.get("max_tokens_setting", "budget")
        
        # New: Inference Strategy 
        # "interactive" = ltthinker agentic loop
        # "vanilla_cot" = standard zero-shot CoT
        self.inference_type = kwargs.get("inference_type", "interactive") 

        # Interactive specific settings (only used if inference_type="interactive")
        self.max_turns = kwargs.get("max_turns", 10)
        
        # Ablation settings
        self.disable_expand = kwargs.get("disable_expand", False)
        self.disable_fold = kwargs.get("disable_fold", False)
        
        self.seed = list(kwargs.get("seed", [0]))
        self.system_prompt = kwargs.get("system_prompt", False)
        
        # Model settings
        self.reasoning_model_path = kwargs.get("reasoning_model", "Qwen/Qwen2.5-7B-Instruct")        
        self.temperature = kwargs.get("temperature", 0.0)
        self.top_p = kwargs.get("top_p", 1.0)
        self.presence_penalty = kwargs.get("presence_penalty", 0.0)
        self.repetition_penalty = kwargs.get("repetition_penalty", 1.05)

        # mode flags
        mode = kwargs.get("mode", "hf")
        self.use_vllm = (mode == "vllm")
        self.use_openai = (mode == "openai")
        self.use_hf = (mode == "hf")        
        
        # VLLM / OpenAI specific setup ... (Keep your original code here)
        self.tokenizer_path = kwargs.get("tokenizer_path", self.reasoning_model_path)
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.7)
        self.api_key = kwargs.get("api_key", "EMPTY")
        self.base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        self.batch_size = kwargs.get("batch_size", 1)

        self.scratchpad = ScratchPad()
        
        output_file = kwargs.get("output_file")
        if output_file is None:
            output_dir = kwargs.get("output_dir", "results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include type in filename
            if self.end_idx is not None:
                output_file = f"{output_dir}/{self.dataset_name}_{self.inference_type}_results_{self.start_idx}_{self.end_idx}_{timestamp}"
            else:
                output_file = f"{output_dir}/{self.dataset_name}_{self.inference_type}_results_{timestamp}"
        self.output_file = output_file + ".json"
        self.output_file_jsonl = output_file + ".jsonl"
        print(f"Results will be saved to: {self.output_file}")
        if self.disable_expand: print("INFO: Expand action is DISABLED.")
        if self.disable_fold: print("INFO: Fold action is DISABLED.")

        # Stop words logic
        # If vanilla CoT, we usually don't want XML tag stop words, just EOS.
        if self.inference_type == "interactive":
            self.stop_words = ["</commit>", "</expand>", "</fold>", "</answer>"]
        else:
            self.stop_words = [] # Let the model run until EOS

        # Initialization
        if self.use_vllm:
            if not VLLM_AVAILABLE: raise ImportError("vllm missing")
            self._init_vllm()
        elif self.use_hf:
            self._init_transformers()
        elif self.use_openai:
            if not OPENAI_AVAILABLE: raise ImportError("openai missing")
            self._init_openai()
        else:
            raise ValueError("Set use_vllm=True or use_openai=True or use_hf=True")

    def _init_vllm(self):
        print("Initializing Tokenizer and VLLM model...")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path)
            if "llama" in self.reasoning_model_path.lower():
                self.model = LLM(
                    model=self.reasoning_model_path,
                    tokenizer=self.tokenizer_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    max_model_len=131072,
                )
            else:
                self.model = LLM(
                    model=self.reasoning_model_path,
                    tokenizer=self.tokenizer_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    dtype=torch.bfloat16,
                    enable_prefix_caching=True,
                    trust_remote_code=True,
                )                
            # stop_words = ["</commit>", "</expand>", "</fold>", "</answer>"]
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                repetition_penalty=self.repetition_penalty,
                max_tokens=self.max_tokens,
                stop=self.stop_words,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [],
            )
            print("Tokenizer and VLLM model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize VLLM components: {e}", exc_info=True)
            raise

    def _init_openai(self):
        print("Initializing OpenAI Client...")
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=600.0,
                max_retries=5
            )

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path)            
            self.stop_words = ["</commit>", "</expand>", "</fold>", "</answer>"]
            
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI Client: {e}", exc_info=True)
            raise

    def _call_openai_worker(self, messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
            """ OpenAI API Worker"""
            start_time = time.time()
            try:                                                 
                response = self.client.chat.completions.create(
                    model=self.reasoning_model_path,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,                    
                    stop=self.stop_words,
                    extra_body={
                        "repetition_penalty": self.repetition_penalty
                    }
                )
                end_time = time.time()

                return {
                    "text": response.choices[0].message.content,
                    "in_len": response.usage.prompt_tokens,
                    "out_len": response.usage.completion_tokens,
                    "latency": end_time - start_time,
                    "error": None
                }
            except Exception as e:
                logging.error(f"OpenAI API Call Failed: {e}")                
                end_time = time.time()
                return {
                    "text": "",
                    "in_len": 0,
                    "out_len": 0,
                    "latency": end_time - start_time,
                    "error": str(e)
                }

    def _init_transformers(self):
            print("Initializing Transformers (Native) model...")
            try:
                # Determine device
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_path)
                
                # Load model
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.reasoning_model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.model.eval()
                
                # Define stop words for manual truncation later
                self.stop_words = ["</commit>", "</expand>", "</fold>", "</answer>"]
                print(f"Transformers model initialized on {self.device}.")
            except Exception as e:
                logging.error(f"Failed to initialize Transformers components: {e}", exc_info=True)
                raise

    # --- Abstract Methods ---
    def load_questions(self) -> List[Dict[str, Any]]:
        """load questions from dataset."""
        raise NotImplementedError

    def format_case(self, data: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        process a single case from the dataset.
        Returns:
            question_text: text of the question
            options_text: text of the options (e.g. "A) ... B) ..."), can be empty for math questions
            correct_answer: standardized answer (e.g. "A" or "42")
        """
        raise NotImplementedError

    def generate_responses_batched(self, batch_size: int = 16) -> Tuple[Dict[str, Any], float]:
        """Main VLLM Batched Execution Loop."""
        raw_questions = self.load_questions()
        print(f"====== Loaded {len(raw_questions)} questions ======")
        print(f"====== Mode: {self.inference_type.upper()} (Batched VLLM) ======")

        all_results = []
        accumulated_tokenize_time = 0        
        global_pbar = tqdm(total=len(raw_questions), desc="Total Progress")
        for start in range(0, len(raw_questions), batch_size):
            end = min(start + batch_size, len(raw_questions))
            batch_questions = raw_questions[start:end]            

            # 1. Initialize State for all questions in the batch
            states = []
            for i, row in enumerate(batch_questions):
                try:
                    q_text, opt_text, ans = self.format_case(row)
                    state = {
                        "id": i,
                        "original_row": row,
                        "question": q_text,
                        "options": opt_text,
                        "answer": ans,
                        "scratchpad": ScratchPad(), # Each case gets its own scratchpad
                        "history_list": [],
                        "messages": [], # Current round messages
                        "turn": 1,
                        "finished": False,
                        "final_reply": "",
                        "stats": {
                            "total_input": 0, 
                            "total_output": 0, 
                            "peak": 0,
                            "cumulative_generation_time": 0.0  # To track total generation time across turns for this case
                        },
                        "start_time": datetime.now(),
                        "max_length": 0, # For interactive mode, we can track the max prompt length to adjust generation parameters if needed
                        "last_op": None, # To track the last operation type (commit/expand/fold) for potential adaptive strategies
                        "consecutive_expand_or_fold_cnt": 0, # To track if we have too many consecutive expand/fold actions without a commit or final answer, which might indicate a parsing issue
                        "consecutive_no_xml_tag_cnt": 0 # To track if we have too many consecutive generations without any XML tags
                    }
                    
                    # Initial Prompt Setup
                    if self.inference_type == "vanilla_cot":
                        # Vanilla CoT Setup
                        if self.dataset_name == "gsm8k": prompt_t = PROMPT_VANILLA_GSM8K
                        elif self.dataset_name == "mmlu": prompt_t = PROMPT_VANILLA_MMLU
                        elif self.dataset_name == "gpqa": prompt_t = PROMPT_VANILLA_GPQA
                        elif self.dataset_name == "bbh": prompt_t = PROMPT_VANILLA_BBH
                        else: prompt_t = "{problem}\n{options}"
                        
                        content = prompt_t.replace("{problem}", q_text).replace("{options}", opt_text)
                        msgs = [{"role": "user", "content": content}]
                        if self.system_prompt:
                            msgs = [{"role": "system", "content": SYSTEM_PROMPT_VANILLA}] + msgs
                        state["messages"] = msgs
                    
                    elif self.inference_type == "token_skip":
                        if self.dataset_name == "gsm8k": prompt_t = PROMPT_VANILLA_GSM8K
                        elif self.dataset_name == "mmlu": prompt_t = PROMPT_VANILLA_MMLU
                        elif self.dataset_name == "gpqa": prompt_t = PROMPT_VANILLA_GPQA
                        elif self.dataset_name == "bbh": prompt_t = PROMPT_VANILLA_BBH
                        else: prompt_t = "{problem}\n{options}"
                        content = prompt_t.replace("{problem}", q_text).replace("{options}", opt_text) + f"<|eot_id|>0.5<|eot_id|>"                        
                        msgs = [{"role": "user", "content": content}]
                        if self.system_prompt:
                            msgs = [{"role": "system", "content": SYSTEM_PROMPT_VANILLA}] + msgs
                        state["messages"] = msgs

                    else:
                        # Interactive Initial Setup
                        if self.dataset_name in ['gsm8k', 'bbh']:
                            prompt_t = PROMPT_OPEN_ENDED
                        else:
                            prompt_t =  PROMPT_MCQ
                        
                        content = prompt_t.replace("{problem}", q_text).replace("{options}", opt_text)                        
                        msgs = [{"role": "user", "content": content}]
                        if self.system_prompt:
                            msgs = [{"role": "system", "content": SYSTEM_PROMPT_LTTHINKER}] + msgs
                        state["messages"] = msgs
                        text_prompt = self.tokenizer.apply_chat_template(
                            state["messages"], tokenize=False, add_generation_prompt=True
                        )
                        state["max_length"] = len(self.tokenizer.encode(text_prompt)) + self.max_tokens

                    states.append(state)
                except Exception as e:
                    logging.error(f"Formatting error at idx {i}: {e}")
                    continue

            active_indices = list(range(len(states)))
            
            # 2. Main Generation Loop
            # For Vanilla CoT, this runs once. For Interactive, it loops until max_turns or all finished.                    
                                    
            while active_indices:
                # Prepare batch prompts
                batch_prompts = []
                batch_sampling_params = []
                            
                batch_messages = []
                batch_max_tokens = []

                batch_map = [] # maps batch_idx -> state_idx           
                
                for idx in active_indices:
                    state = states[idx]

                    if state["consecutive_expand_or_fold_cnt"] >= 4:                        
                        # We can choose to set finished=True here to force it out of the active loop, or we can just rely on the warning and let it continue. For now, we'll just log the warning and let it continue.
                        state["finished"] = True
                        logging.warning(f"State idx {idx} has {state['consecutive_expand_or_fold_cnt']} consecutive expand/fold actions. This may indicate a parsing issue. Forcing final answer generation in the next turn.")
                        global_pbar.update(1)
                        continue

                    text_prompt = self.tokenizer.apply_chat_template(
                        state["messages"], tokenize=False, add_generation_prompt=True
                    )
                    max_model_len = state.get("max_length", 0)
                    
                    if self.inference_type == "interactive":
                        start_time = datetime.now()
                        # max_tokens = max_model_len - len(self.tokenizer.encode(text_prompt))
                        if self.max_tokens_setting == "budget":
                            turn_max_tokens = max_model_len
                        elif self.max_tokens_setting == "throughput":
                            turn_max_tokens = self.turn_max_tokens 
                        max_tokens = min(turn_max_tokens, max_model_len - len(self.tokenizer.encode(text_prompt)))
                        end_time = datetime.now()
                        accumulated_tokenize_time += (end_time - start_time).total_seconds()    
                    else:
                        max_tokens = self.max_tokens   

                    if max_tokens <= 0:
                        state["finished"] = True
                        logging.warning(f"Prompt too long for model context at idx {idx}. Marking as finished without generation.")
                        global_pbar.update(1)                        
                        continue

                    if self.use_vllm:
                        sampling_params = SamplingParams(
                            temperature=self.temperature,
                            top_p=self.top_p,
                            presence_penalty=self.presence_penalty,
                            repetition_penalty=self.repetition_penalty,
                            max_tokens=max_tokens,
                            # max_tokens=self.max_tokens,
                            stop=self.stop_words,
                            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else [],
                        )
                        batch_prompts.append(text_prompt)
                        batch_sampling_params.append(sampling_params)

                    elif self.use_openai:
                        batch_messages.append(state["messages"])
                        batch_max_tokens.append(max_tokens)
                
                    batch_map.append(idx)

                if len(batch_map) == 0:
                    # No active prompts to process (e.g. all had token limit issues), break the loop
                    break
                    
                parsed_outputs =[]
                
                if self.use_vllm:
                    vllm_start_t = time.time()
                    outputs = self.model.generate(batch_prompts, batch_sampling_params, use_tqdm=False)
                    vllm_end_t = time.time()
                    for out in outputs:
                        parsed_outputs.append({
                            "text": out.outputs[0].text.strip(),
                            "in_len": len(out.prompt_token_ids),
                            "out_len": len(out.outputs[0].token_ids),
                            "latency": vllm_end_t - vllm_start_t # not using it for now since vllm doesn't give us per-sample generation time, but we can consider it as an upper bound for the whole batch
                        })
                
                elif self.use_openai:                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                        futures =[
                            executor.submit(self._call_openai_worker, msgs, m_tokens)
                            for msgs, m_tokens in zip(batch_messages, batch_max_tokens)
                        ]
                        for future in futures:
                            res = future.result()
                            parsed_outputs.append({
                                "text": res["text"].strip() if res["text"] else "",
                                "in_len": res["in_len"],
                                "out_len": res["out_len"],
                                "latency": res["latency"]
                            })

                # Process Outputs
                next_active_indices = []
                
                for batch_i, output in enumerate(parsed_outputs):
                    state_idx = batch_map[batch_i]
                    state = states[state_idx]
                    
                    generated_text = output["text"]
                    logging.info("Model generated text:\n" + generated_text)
                    
                    # Token Tracking
                    in_len = output["in_len"]
                    out_len = output["out_len"]
                    step_latency = output["latency"]
                    state["stats"]["total_input"] += in_len
                    state["stats"]["total_output"] += out_len
                    state["stats"]["cumulative_generation_time"] += step_latency
                    current_total = in_len + out_len
                    if current_total > state["stats"]["peak"]:
                        state["stats"]["peak"] = current_total

                    prompt_log = json.dumps(state["messages"], ensure_ascii=False)
                    
                    # --- Branching Handling ---
                    
                    if self.inference_type == "vanilla_cot" or self.inference_type == "token_skip":
                        # Vanilla is always 1-turn
                        state["final_reply"] = generated_text
                        state["history_list"].append({
                            "turn": 1, "prompt": prompt_log, "generated_text": generated_text,
                            "input_tokens": in_len, "output_tokens": out_len
                        })
                        state["finished"] = True
                        global_pbar.update(1) 
                    
                    else:
                        # Interactive Logic
                        # Log history
                        state["history_list"].append({
                            "turn": state["turn"],
                            "prompt": prompt_log,
                            "generated_text": generated_text,
                            "input_tokens": in_len,
                            "output_tokens": out_len
                        })

                        # Parse output
                        try:
                            output_type, content, is_success = parse_model_output(generated_text)
                        except:
                            output_type, content, is_success = "error", None, False
                        
                        is_finished = False
                        if not is_success:
                            state["consecutive_no_xml_tag_cnt"] += 1
                            logging.warning(f"No XML tags detected in generated text for state idx {state_idx}. Consecutive no XML tag count: {state['consecutive_no_xml_tag_cnt']}")
                        else:
                            state["consecutive_no_xml_tag_cnt"] = 0 # reset counter on successful parse

                        if output_type == "final_answer":
                            if content and "final_answer" in content:
                                state["final_reply"] = content["final_answer"]
                            else:
                                state["final_reply"] = generated_text
                            is_finished = True

                        elif output_type == "commit":
                            detailed_reasoning = content.get("detailed_reasoning", "")
                            summary = content.get("summary", "")
                            state["scratchpad"].add_step(detailed_reasoning, summary)
                        
                        elif output_type == "expand":
                            if self.disable_expand:
                                logging.info("Expand action detected but disabled. Skipping scratchpad update.")
                                # continue
                            else:
                                id = content.get("step_id")
                                if id is not None and id != '':
                                    try: 
                                        state["scratchpad"].expand_step(id)
                                        logging.info(f"Expanded step ID: {id}")
                                    except IndexError:
                                        logging.warning(f"Invalid step_id {id} for expand command. Retrying...")
                                        continue
                                else:
                                    logging.warning("No step_id provided for expand command.")

                        elif output_type == "fold":
                            if self.disable_fold:
                                logging.info("Fold action detected but disabled. Skipping scratchpad update.")
                                # continue
                            else:
                                id = content.get("step_id")
                                if id is not None and id != '':
                                    try: 
                                        state["scratchpad"].fold_step(id)
                                        logging.info(f"Folded step ID: {id}")
                                    except IndexError:
                                        logging.warning(f"Invalid step_id {id} for fold command. Retrying...")
                                        continue
                                else:
                                    logging.warning("No step_id provided for fold command.")
                        else:
                            logging.warning("Unrecognized output type or no action taken.")                            
                        
                        # Check limits
                        state["turn"] += 1
                        if state["last_op"] in ["expand", "fold"] and output_type in ["expand", "fold"]:
                            state["consecutive_expand_or_fold_cnt"] += 1
                        else:
                            state["consecutive_expand_or_fold_cnt"] = 0

                        state["last_op"] = output_type
                        
                        if not is_finished:
                            if state["turn"] > self.max_turns:
                                # FORCE ANSWER LOGIC
                                history_text = state["scratchpad"].build_history_text_for_prompt()
                                if self.dataset_name in ['gsm8k', 'bbh']:
                                    prompt_t = PROMPT_OPEN_ENDED_2
                                else:
                                    prompt_t = PROMPT_MCQ_2
                                
                                base = prompt_t.replace("{problem}", state["question"]).replace("{options}", state["options"]).replace("{scratchpad}", history_text)
                                force_instruction = "\n\nSYSTEM NOTIFICATION: MAX STEPS REACHED. OUTPUT FINAL ANSWER NOW."
                                state["messages"] = [{"role": "user", "content": base + force_instruction}]
                                if self.system_prompt:
                                    state["messages"] = [{"role": "system", "content": SYSTEM_PROMPT_LTTHINKER}] + state["messages"]
                                
                                # Next iteration will be the forced generation
                                # We treat it as one last "active" turn, but we might want a special flag
                                # Actually, just letting it loop one more time with this prompt is easiest.
                                # However, to prevent infinite loops if model refuses, we can set a flag "forcing=True"
                                if state.get("forcing"):
                                    # Already forced once, just stop
                                    state["final_reply"] = generated_text # whatever we got
                                    is_finished = True
                                else:
                                    state["forcing"] = True
                                    next_active_indices.append(state_idx)
                                    continue

                            else:
                                # Prepare next turn prompt
                                history_text = state["scratchpad"].build_history_text_for_prompt()
                                if self.dataset_name in ['gsm8k', 'bbh']:
                                    prompt_t = PROMPT_OPEN_ENDED_2
                                else:
                                    prompt_t = PROMPT_MCQ_2
                                
                                content = prompt_t.replace("{problem}", state["question"]).replace("{options}", state["options"]).replace("{scratchpad}", history_text)
                                next_msgs = [{"role": "user", "content": content}]
                                if self.system_prompt:
                                    next_msgs = [{"role": "system", "content": SYSTEM_PROMPT_LTTHINKER}] + next_msgs
                                state["messages"] = next_msgs
                                next_active_indices.append(state_idx)

                        if is_finished:
                            state["finished"] = True
                            global_pbar.update(1) 

                active_indices = next_active_indices

            # 3. Compile Results
            for state in states:
                row = state["original_row"]
                row["processed_question"] = state["question"]
                row["processed_options"] = state["options"]
                row["answer"] = state["answer"]
                
                row["model_full_history"] = state["history_list"]
                row["model_final_output"] = state["final_reply"]
                row["model_final_round_messages"] = state["messages"]
                row["model_answer"] = self.extract_answer(state["final_reply"])
                
                end_time = datetime.now()            

                if self.use_vllm:
                    row["inference_time_seconds"] = (end_time - state["start_time"]).total_seconds()
                elif self.use_openai:
                    row["inference_time_seconds"] = state["stats"]["cumulative_generation_time"] # OpenAI mode tracks generation time directly from API calls


                # if self.use_openai:
                #     # Use hours as unit, rounded to 2 decimal places                    
                #     row["cumulative_generation_time_hours"] = round(state["stats"]["cumulative_generation_time"] / 3600.0, 2)
                
                row["total_input_tokens"] = state["stats"]["total_input"]
                row["total_output_tokens"] = state["stats"]["total_output"]
                row["peak_token_num"] = state["stats"]["peak"]
                row["max_model_len"] = state.get("max_length", 0)
                
                all_results.append(row)
                self.save_incremental_result_jsonl(row)

        global_pbar.close()
        return {"examples": all_results}, accumulated_tokenize_time

    def save_incremental_result_jsonl(self, example: Dict[str, Any]) -> None:
        def _to_native(x):
            if isinstance(x, np.integer):
                return int(x)
            if isinstance(x, np.floating):
                return float(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        def _convert(obj):
            if isinstance(obj, dict):
                return {k: _convert(_to_native(v)) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(_to_native(v)) for v in obj]
            return _to_native(obj)

        record = _convert(example)

        out_path = getattr(self, "output_file_jsonl", None) or (self.output_file + ".jsonl")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def extract_answer(self, s):
        pattern = r'\\boxed\{([^{}]*|\{[^{}]*\})*\}'
        # pattern = r'(?:\\|)?(?:b|B)?oxed\s*\{((?:.|\n)*?)\}'
        
        def match_balanced_braces(text):
            stack = []
            result = []
            for char in text:
                if char == '{':
                    stack.append(char)
                elif char == '}':
                    if stack:
                        stack.pop()
                    else:
                        return None  
                result.append(char)
            return ''.join(result) if not stack else None
        
        matches = re.finditer(pattern, s)
        results = []
        for match in matches:
            content = match.group(0)[7:-1]  
            balanced_content = match_balanced_braces(content)
            if balanced_content:
                results.append(balanced_content)
        
        if len(results) == 0:
            # return "error"
            search_str = "oxed{"
            start_idx = s.find(search_str)
            
            if start_idx == -1:
                return "error"
            
            content_start = start_idx + len(search_str)
            
            balance = 1
            end_idx = content_start
            
            while end_idx < len(s) and balance > 0:
                char = s[end_idx]
                if char == '{':
                    balance += 1
                elif char == '}':
                    balance -= 1
                
                if balance == 0:
                    break
                end_idx += 1

            if balance != 0:
                return "error"

            return s[content_start: end_idx]
        else:
            return results[-1].strip()

    def _digit_compare(self, model_ans: str, gt_ans: str):
        """
        """
        def normalize(s):        
            return s.replace("\\", "").replace("$", "").replace(",", "").replace("%", "").strip()

        m_clean = normalize(model_ans)
        g_clean = normalize(gt_ans)

        if not m_clean or not g_clean:
            return False

        for item in [m_clean, g_clean]:
            for ch in item:
                if ch not in "0123456789.-+":
                    return False
                    
        try:
            return abs(float(m_clean) - float(g_clean)) < 1e-9
        except:
            return False
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Default evaluation: Exact Match."""
        if not results or "examples" not in results: return {}
        examples = results["examples"]
        num_questions = len(examples)
        if num_questions == 0: return {"accuracy_avg": 0.0}

        solved = 0
        peak_token_num = 0
        for ex in examples:
            gold = str(ex.get("answer", "")).strip().lower()
            pred = str(ex.get("model_answer", "")).strip().lower()
            if ex.get("peak_token_num", 0) > peak_token_num:
                peak_token_num = ex.get("peak_token_num", 0)
            
            # Simple cleanup comparison
            if gold == pred:
                solved += 1
            # Special handling for each dataset
            elif self.dataset_name in ['gpqa', 'mmlu']:
                # MCQ: extract first letter A-E
                pattern = r'^([A-E])[\).]?'
                gold_match = re.match(pattern, gold, re.IGNORECASE)
                pred_match = re.match(pattern, pred, re.IGNORECASE)
                if gold_match and pred_match:
                    if gold_match.group(1).upper() == pred_match.group(1).upper():
                        solved += 1
            elif self.dataset_name == "gsm8k":
                digit_comp = self._digit_compare(pred, gold)
                if digit_comp is True:
                    solved += 1

            elif self.dataset_name == "bbh":
                matched = False
                # First try MCQ style
                pattern = r'^([A-E])[\).]?'
                gold_match = re.match(pattern, gold, re.IGNORECASE)
                pred_match = re.match(pattern, pred, re.IGNORECASE)
                if gold_match and pred_match:
                    if gold_match.group(1).upper() == pred_match.group(1).upper():
                        solved += 1
                        matched = True
                if not matched:
                    digit_comp = self._digit_compare(pred, gold)
                    if digit_comp is True:
                        solved += 1

        accuracy = solved / num_questions
        return {"num_total": num_questions, "num_solved": solved, "accuracy_avg": accuracy, "peak": peak_token_num}

    def eval(self):
            eval_start = datetime.now()

            results_data, accumulated_tokenize_time = self.generate_responses_batched(batch_size=self.batch_size)

            eval_end = datetime.now()
            total_time_seconds_wall = (eval_end - eval_start).total_seconds() - accumulated_tokenize_time                        

            final_stats = self.evaluate_responses(results_data)

            examples = results_data.get("examples", [])
            num_examples = len(examples)

            total_time_seconds_sum_examples = float(sum(ex.get("inference_time_seconds", 0.0) for ex in examples))
            avg_sec_per_sample = (total_time_seconds_wall / num_examples) if num_examples > 0 else None

            output_data_to_save = {
                "examples": examples,
                "results": final_stats,
                "total_time_seconds_wall": total_time_seconds_wall,
                "avg_sec_per_sample": avg_sec_per_sample,

                "total_time_seconds_sum_examples": total_time_seconds_sum_examples,
                "accumulated_tokenize_time": accumulated_tokenize_time,
            }
            
            with open(self.output_file, "w", encoding='utf-8') as f: 
                json.dump(output_data_to_save, f, indent=4, ensure_ascii=False)
            print(f"\nEvaluation complete. Results saved to: {self.output_file}")
            if final_stats:
                print(
                    f"Final Accuracy: {final_stats.get('accuracy_avg', 0.0):.4f}, "
                    f"Wall Time: {total_time_seconds_wall:.2f}s, "
                    f"Avg Sec/Sample (wall): {avg_sec_per_sample:.4f}s"
                )
# ==========================================
# Concrete Subclasses
# ==========================================

class GPQABenchmark(BaseInteractiveBenchmark):
    def __init__(self, **kwargs):
        kwargs["dataset_name"] = "gpqa"
        super().__init__(**kwargs)

    def load_questions(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(DATA_DIR, "gpqa.json")
        print(f"Loading GPQA from local file: {file_path}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {file_path}. Please check the path.")

        target_subset = self.subset_name if self.subset_name else 'all'
        
        all_questions = []
        for domain, items in raw_data.items():
            if target_subset == 'all' or domain == target_subset:
                all_questions.extend(items)

        if self.debug:
            all_questions = all_questions[:self.debug_num]
        
        print(f"Total questions loaded: {len(all_questions)}")
        return all_questions

    def format_case(self, data: Dict[str, Any]) -> Tuple[str, str, str]:
        question_text = data.get("pure_question", data.get("question", "")).strip()
        
        choices = data.get("choice_list", [])
        gold_answer_raw = str(data.get("answer", "")).strip()
        
        correct_answer_text = ""
        if len(gold_answer_raw) == 1 and gold_answer_raw.upper() in ['A', 'B', 'C', 'D']:
            idx = ord(gold_answer_raw.upper()) - ord('A')
            if 0 <= idx < len(choices):
                correct_answer_text = choices[idx]
        else:
            correct_answer_text = gold_answer_raw

        shuffled_choices = choices[:]
        random.shuffle(shuffled_choices)
        
        options_map = ["A", "B", "C", "D"]
        current_options = shuffled_choices[:4]
        
        options_to_answers = dict(zip(options_map, current_options))
        mc_string = " ".join(f"{k}) {v.strip()}" for k, v in options_to_answers.items())
        
        correct_letter = ""
        for k, v in options_to_answers.items():
            if v == correct_answer_text:
                correct_letter = k
                break
        
        if not correct_letter and len(gold_answer_raw) == 1:
             correct_letter = gold_answer_raw

        return question_text, mc_string, correct_letter


class MMLUBenchmark(BaseInteractiveBenchmark):
    def __init__(self, **kwargs):
        kwargs["dataset_name"] = "mmlu"
        super().__init__(**kwargs)

    def load_questions(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(DATA_DIR, "mmlu.json")
        print(f"Loading MMLU from local file: {file_path}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {file_path}.")

        target_subset = self.subset_name if self.subset_name else 'all'
        
        all_questions = []
        for domain, items in raw_data.items():
            if target_subset == 'all' or target_subset in domain:
                all_questions.extend(items)

        if self.debug: 
            all_questions = all_questions[:self.debug_num]

        print(f"Total questions loaded: {len(all_questions)}")
        return all_questions

    def format_case(self, data: Dict[str, Any]) -> Tuple[str, str, str]:
        
        q_text = data.get('question', "").strip()
        choices = data.get('choices_list', [])
        labels = ["A", "B", "C", "D"]
        
        options_str_list = []
        for label, text in zip(labels, choices):
            options_str_list.append(f"{label}) {text.strip()}")
        options_text = "\n".join(options_str_list)
        
        ans_letter = data.get('answer', "").strip()
            
        return q_text, options_text, ans_letter


class GSM8KBenchmark(BaseInteractiveBenchmark):
    def __init__(self, **kwargs):
        kwargs["dataset_name"] = "gsm8k"
        super().__init__(**kwargs)

    def load_questions(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(DATA_DIR, "gsm8k.json")
        print(f"Loading GSM8K from local file: {file_path}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {file_path}.")

        all_questions = []
        for key in raw_data:
            all_questions.extend(raw_data[key])

        if self.debug: 
            all_questions = all_questions[:self.debug_num]
        return all_questions

    def format_case(self, data: Dict[str, Any]) -> Tuple[str, str, str]:
        q = data['question']
        gold_val = str(data['answer']).strip()
        
        if "####" in gold_val:
            gold_val = gold_val.split("####")[-1].strip()
        
        options_text = "(Please put your final numerical answer in \\boxed{}.)"
        return q, options_text, gold_val
    
    def extract_answer(self, text: str) -> str:
        boxed = super().extract_answer(text)
        if boxed: return boxed
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            return nums[-1]
        return ""


class BBHBenchmark(BaseInteractiveBenchmark):
    def __init__(self, **kwargs):
        kwargs["dataset_name"] = "bbh"
        super().__init__(**kwargs)

    def load_questions(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(DATA_DIR, "bbh.json")
        print(f"Loading BBH from local file: {file_path}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {file_path}.")

        target_subset = self.subset_name if self.subset_name else 'all'
        
        all_questions = []
        for domain, items in raw_data.items():
            if target_subset == 'all' or domain == target_subset:
                all_questions.extend(items)

        if self.debug: 
            all_questions = all_questions[:self.debug_num]
            
        print(f"Total questions loaded: {len(all_questions)}")
        return all_questions

    def format_case(self, data: Dict[str, Any]) -> Tuple[str, str, str]:
        # root level: 'question', 'answer', 'domain'
        # meta_data: 'input', 'target'
        
        q_text = data.get('question', "")
        
        if not q_text and 'meta_data' in data:
             q_text = data['meta_data'].get('input', "")

        target = data.get('answer', "")
        
        options_text = ""
        return q_text, options_text, target

class GeneralReasoningDataset(BaseInteractiveBenchmark):
    def __init__(self, **kwargs):
        kwargs["dataset_name"] = "general_reasoning"
        super().__init__(**kwargs)

    def load_questions(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(DATA_DIR, "general_reasoning_ids_all.jsonl")
        print(f"Loading general reasoning data from local file: {file_path}...")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # raw_data = json.load(f)
                # load jsonl
                raw_data = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {file_path}.")

        if self.end_idx is None:
            self.end_idx = len(raw_data)
        all_questions = raw_data[self.start_idx: self.end_idx]  # Assuming the JSON is a list of questions

        if self.debug: 
            all_questions = all_questions[:self.debug_num]
        return all_questions

    def format_case(self, data: Dict[str, Any]) -> Tuple[str, str, str]:
        # New data structure: 'problem', 'answer'
        q = data['problem']
        gold_ans = str(data['answer']).strip()
        
        options_text = "(Please provide your answer in \\boxed{}.)"
        return q, options_text, gold_ans
# ==========================================
# Factory
# ==========================================

def run_benchmark(**kwargs):
    if kwargs.get("dataset_name") == "gpqa":
        bench = GPQABenchmark(**kwargs)
    elif kwargs.get("dataset_name") == "mmlu":
        bench = MMLUBenchmark(**kwargs)
    elif kwargs.get("dataset_name") == "gsm8k":
        bench = GSM8KBenchmark(**kwargs)
    elif kwargs.get("dataset_name") == "bbh":
        bench = BBHBenchmark(**kwargs)
    elif kwargs.get("dataset_name") == "general_reasoning":
        bench = GeneralReasoningDataset(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset_name: {kwargs.get('dataset_name')}")
    
    bench.eval()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":

    params = {
        "reasoning_model": "/data/PLMs/Qwen2.5-7B-Instruct",
        
        # Switch flags here:
        "mode": "vllm",  # "vllm", "openai", "hf"
        "inference_type": "vanilla_cot",  # or "interactive"
        
        "debug": True,
        "debug_num": 5,
        "dataset_name": "gpqa", # mmlu, gsm8k, bbh
        
        # Interactive settings (ignored in vanilla_cot)
        "max_turns": 60,
        "system_prompt": True,
        
        # Output settings
        "output_dir": "./benchmark_results",
        
        # Generation Params
        "temperature": 0.0, # Usually 0 for standard CoT benchmark
        "top_p": 1.0,

        # Ablation flags
        "disable_expand": False,
        "disable_fold": False,
    }
    
    run_benchmark(**params)
