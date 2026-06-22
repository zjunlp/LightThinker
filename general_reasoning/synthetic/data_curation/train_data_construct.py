import json
import os
import re
from difflib import SequenceMatcher
from collections import defaultdict
import argparse
import random


def extract_reasoning_text(hist, base_id=None, round_id=None, log_file="missing_reasoning.log"):    
    output = hist.get("output", {}) or {}
    raw_resp = output.get("raw_response", {}) or {}
    raw_content = raw_resp.get("content", "")
    if raw_content:
        return raw_content

    reason = output.get("reasoning", "")
    if reason:
        return reason

    try:
        log_entry = {
            "base_id": base_id,
            "round": round_id,
            "hist": hist,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[extract_reasoning_text] failed to log missing reasoning for id={base_id}: {e}")
    return ""

def load_jsonl(filepath, max_lines=None):
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line {i+1} in {filepath}")
                    with open("bad_lines.log", "a") as log_f:
                        log_f.write(f"{filepath}: line {i+1}\n")
                    continue

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"[load_json] JSON : {e}")
            return None

def replace_boxed(text, new_content):
    search_str = "\\boxed{"
    start_idx = text.rfind(search_str)
    if start_idx == -1:
        return None
    content_start = start_idx + len(search_str)
    balance = 1
    end_idx = content_start
    while end_idx < len(text) and balance > 0:
        char = text[end_idx]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
        if balance == 0:
            break
        end_idx += 1
    if balance != 0:
        print(f"Warning: Unmatched braces in text: {text}")
        return None
    return text[:content_start] + str(new_content) + text[end_idx:]

def extract_last_boxed(text):
    search_str = "\\boxed{"
    start_idx = text.rfind(search_str)
    if start_idx == -1:
        return None
    content_start = start_idx + len(search_str)
    balance = 1
    end_idx = content_start
    while end_idx < len(text) and balance > 0:
        char = text[end_idx]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1
        if balance == 0:
            break
        end_idx += 1
    if balance != 0:
        return None
    return text[content_start:end_idx]

def fix_instr(instr):
    instr = re.sub(r'^\s*Problem:\s*', '', instr, count=1)
    instr = re.sub(r"\n+Let's think step by step\.\s*$", '', instr)
    return instr

def calculate_similarity(text1, text2):
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()

def extract_remaining_samples(original_file, remaining_ids_file, output_file):    
    if not os.path.exists(remaining_ids_file) or not os.path.exists(original_file):
        return
        
    with open(remaining_ids_file, 'r') as f:
        remaining_ids = set(line.strip() for line in f if line.strip())
    
    samples = []
    with open(original_file, 'r') as f:
        data = load_jsonl(original_file)
        for sample in data:
            if 'id' in sample and str(sample['id']) in remaining_ids:
                samples.append(sample)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as out_f:
        for sample in samples:
            out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"Extracted {len(samples)} remaining samples to {output_file}")


# ==========================================
# Core Logic: Unified Validation
# ==========================================

def validate_trajectory(obj, valid_ids_set, filtering=False):
    base_id = obj.get("id", "unknown_id")
    histories = obj.get("model_histories", [])
    is_correct = obj.get("is_correct", False)
        
    stats = defaultdict(int)

    if not is_correct:
        return False, "incorrect answer", stats

    if valid_ids_set and base_id not in valid_ids_set:
        return False, "duplicate or unknown id", stats

    if not histories or not histories[0]:
        return False, "no histories", stats

    last_hist = histories[0][-1]
    
    tool_calls = last_hist.get("tool_calls", [])
    if len(tool_calls) == 0 or tool_calls[0].get("tool_name") != "final_answer":
        return False, "no final_answer in last round", stats

    expand_cnt, fold_cnt, commit_cnt = 0, 0, 0
    for hist in histories[0]:
        t_calls = hist.get("tool_calls", [])
        c_name = t_calls[0].get("tool_name") if t_calls else None
        if c_name == "expand": expand_cnt += 1
        if c_name == "fold": fold_cnt += 1
        if c_name == "commit": commit_cnt += 1

    if filtering:
        if expand_cnt >= commit_cnt:
            stats["c6"] = 1
            return False, f"expand count {expand_cnt} >= commit count {commit_cnt}", stats
    
    expanded_ids = set()
    last_op_name = None
    last_op_target = None
    step = 1
    seen_reasonings, seen_summaries = [], []

    for hist in histories[0]:
        round_id = hist.get("round", None)
        t_calls = hist.get("tool_calls", [])

        if len(t_calls) == 0:
            return False, f"no tool_calls in round {round_id}", stats
        if round_id is None:
            return False, f"no round_id", stats
    
        tool_call_success = False
        active_tool_call = None
        for tool_call in t_calls:
            status = tool_call.get("status", "error")
            if status == "success":
                tool_call_success = True
                active_tool_call = tool_call
                break 
        
        if not tool_call_success:
            return False, f"no successful tool call in round {round_id}", stats

        call_name = active_tool_call.get("tool_name")
        args_dict = active_tool_call.get("arguments", {}) or {}
        target_id = args_dict.get("step_id")
        if target_id is not None:
            target_id = str(target_id)

        if filtering:            
            if call_name in ["expand", "fold"]:
                if target_id and call_name == last_op_name and target_id == last_op_target:
                    stats["c1"] = 1
                    return False, f"consecutive {call_name} on {target_id}", stats
            
            if call_name == "fold":
                if target_id not in expanded_ids:
                    stats["c2"] = 1
                    return False, f"fold on unexpanded id {target_id}", stats

        if call_name == "expand" and target_id:
            expanded_ids.add(target_id)
        
        if filtering and round_id > 2 * step:
            stats["c4"] = 1
            return False, f"step {step} exceeds max_turns", stats
        
        last_op_name = call_name
        last_op_target = target_id
        
        input_msgs = hist.get("input_messages", [])
        if not input_msgs:
            return False, f"no input_messages in round {round_id}", stats
        
        if call_name == "commit":
            detailed_reasoning = args_dict.get("detailed_reasoning", "")
            summary = args_dict.get("summary", "")
            
            if filtering:
                # # Overlap / Duplicate reasoning check
                # is_duplicate = False
                # for prev_reason in seen_reasonings:
                #     if calculate_similarity(detailed_reasoning, prev_reason) >= 0.9:
                #         is_duplicate = True
                #         break
                # for prev_summary in seen_summaries:
                #     if calculate_similarity(summary, prev_summary) >= 0.9:
                #         is_duplicate = True
                #         break
                # if is_duplicate:
                #     stats["c5"] = 1 # Overlap
                #     return False, f"duplicate reasoning in round {round_id}", stats
                                
                if "<｜DSML｜function_calls>" in detailed_reasoning:                     
                     # stats["c7"] = 1
                     pass
            
            seen_reasonings.append(detailed_reasoning)
            seen_summaries.append(summary)
            step += 1

        elif call_name == "final_answer":
            final_answer = args_dict.get("final_answer", "")
            if not final_answer:
                return False, f"empty final_answer in round {round_id}", stats
            
            if "\\boxed{" not in final_answer:
                return False, "no \\boxed{ in final_answer", stats

            # Optional: detailed_reasoning in final answer
            ans_reasoning = args_dict.get("detailed_reasoning", "")
            if ans_reasoning:
                seen_reasonings.append(ans_reasoning)

    return True, None, stats


# ==========================================
# Core Logic: Conversion Modes
# ==========================================

def clean_dsml_tokens(text):
    if not text: return text
    text = re.sub(r"</think>\s*<｜DSML｜function_calls>\s*<｜DSML｜invoke name=\"commit\">\s*<｜DSML｜parameter name=\"detailed_reasoning\" string=\"true\">", "\n\n", text)
    text = re.sub(r"</think>\s*<｜DSML｜function_calls>\s*<｜DSML｜invoke name=\"final_answer\">\s*<｜DSML｜parameter name=\"final_answer\" string=\"true\">", "\n\n", text)
    return text

def convert_stepwise(obj, filtering=False, add_think_tag=False, strict=False, _valid_ids=None):
    samples = []
    base_id = obj.get("id")
    histories = obj.get("model_histories", [])
    model_answers = obj.get("model_answers", [])
    gt = obj.get("answer", "")
    
    gt_replaced_count = 0
    
    seen_reasonings_summaries = []
    abort_flag = False
    for hist in histories[0]:
        round_id = hist.get("round")
        tool_calls = hist.get("tool_calls", [])
        
        active_tool_call = None
        for tool_call in tool_calls:
            if tool_call.get("status", "error") == "success":
                active_tool_call = tool_call
                break
        if not active_tool_call: continue

        call_name = active_tool_call.get("tool_name")
        args_dict = active_tool_call.get("arguments", {}) or {}
        
        # === Instruction (Scratchpad construction) ===
        input_msgs = hist.get("input_messages", [])
        non_system_msgs = [msg["content"] for msg in input_msgs if msg.get("role") != "system"]
        
        question_text = fix_instr(non_system_msgs[0]).strip()
        question_text = re.sub(r'^Return your final response within \\\\?boxed\{\}\.?(\s*|\n*)', '', question_text)

        scratchpad_parts = non_system_msgs[1:-1]
        scratchpad_text = "\n".join(
            re.sub(
                r"\[(Expanded\s+)?Step ID:\s*(\d+)\]",
                lambda m: f"[{m.group(1) or ''}Step {m.group(2)}]",
                msg
            )
            for msg in scratchpad_parts
        ).strip()
        scratchpad_text = re.sub(r"(\[(?:Expanded\s+)?Step \d+\])\s*\n\s*", r"\1 ", scratchpad_text)

        if scratchpad_text:
            instruction = (
                "Return your final response within \\boxed{}.\n"
                "### Question:\n"
                f"{question_text}\n"
                "### Current Scratchpad:\n"
                f"{scratchpad_text}"
            ).strip()
        else:
            instruction = (
                "Return your final response within \\boxed{}.\n"
                "### Question:\n"
                f"{question_text}"
            ).strip()

        # === Output ===
        output = ""
        if call_name == "commit":

        # ==== Insert extra filtering logic =======
            detailed_reasoning = clean_dsml_tokens(args_dict.get("detailed_reasoning", ""))
            summary = args_dict.get("summary", "")

            if filtering:
                is_duplicate = False
                for prev_reason, prev_summary in seen_reasonings_summaries:
                    similarity_reason = calculate_similarity(detailed_reasoning, prev_reason)
                    similarity_summary = calculate_similarity(summary, prev_summary)
                    if similarity_reason >= 0.9 or similarity_summary >= 0.9:
                        is_duplicate = True
                        break
                if is_duplicate: 
                    abort_flag = True
            if add_think_tag:
                output = f"<think>\n{detailed_reasoning.strip()}\n</think>\n\n<commit>{summary}</commit>"
            else:
                output = f"{detailed_reasoning}\n\n<commit>{summary}</commit>"
                

            seen_reasonings_summaries.append((detailed_reasoning, summary))
        
        elif call_name == "expand":
            reason = args_dict.get("reason", "")
            expand_id = args_dict.get("step_id", "")
            if add_think_tag:
                output = f"<think>\n{reason.strip()}\n</think>\n\n<expand>Step {expand_id}</expand>"
            else:
                output = f"{reason.strip()}\n\n<expand>Step {expand_id}</expand>"
        
        elif call_name == "fold":
            reason = args_dict.get("reason", "")
            fold_id = args_dict.get("step_id", "")
            if add_think_tag:
                output = f"<think>\n{reason.strip()}\n</think>\n\n<fold>Step {fold_id}</fold>"    
            else:
                output = f"{reason}\n\n<fold>Step {fold_id}</fold>"
        
        elif call_name == "final_answer":
            raw_final = clean_dsml_tokens(args_dict.get("final_answer", "").strip())
            
            # Replace Boxed Logic
            final_answer_text = replace_boxed(raw_final, str(model_answers[0]).strip())
            answer = extract_last_boxed(final_answer_text)
            if answer is None:
                final_answer_text = raw_final

            if gt and re.match(r'^[A-Za-z]$', str(gt).strip()):
                final_answer_text = replace_boxed(final_answer_text, str(gt).strip())
                gt_replaced_count += 1            
                                  

            # DEFAULT_CHOICES = [
            #     "So now I can conclude the final answer.",
            #     "Now I can provide the final answer.",
            #     "In conclusion, here is the final answer.",
            #     "Therefore, the final answer is as follows.",
            # ]
            # choice = random.choice(DEFAULT_CHOICES)    
            # if add_think_tag:
            #     output = f"<think>\n{choice}\n</think>\n\n<answer>{final_answer_text}</answer>"
            # else:
            #     output += f"{choice}\n\n<answer>{final_answer_text}</answer>\n"

            if add_think_tag:
                output = f"<think>\n{final_answer_text.strip()}\n</think>\n\n" + f"<answer>The final answer is: " + "\\boxed{" + extract_last_boxed(final_answer_text) + "}</answer>"
            else:
                output = f"{final_answer_text}\n\n<answer>The final answer is: " + "\\boxed{" + extract_last_boxed(final_answer_text) + "}</answer>"

        else:
            continue
        
        if abort_flag:
            if strict:
                return [], gt_replaced_count            
            continue
        sample = {
            "id": f"id_{base_id}_round_{round_id}",
            "instruction": instruction,
            "output": output.strip()
        }

        if _valid_ids is not None:
            _valid_ids.add(base_id)
        samples.append(sample)

    return samples, gt_replaced_count

def convert_full_process(obj, filtering=False, add_think_tag=False, _valid_ids=None):
    """
    Mode: Full (Script 2 Logic)
    """
    base_id = obj.get("id")
    if _valid_ids is not None and base_id not in _valid_ids:
        return [], 0
    histories = obj.get("model_histories", [])
    model_answers = obj.get("model_answers", [])
    gt = obj.get("answer", "")
    
    gt_replaced_count = 0
    first_hist = histories[0][0]
    
    # === Instruction ===
    input_msgs = first_hist.get("input_messages", [])
    non_system_msgs = [msg["content"] for msg in input_msgs if msg.get("role") != "system"]
    
    question_text = fix_instr(non_system_msgs[0]).strip()
    question_text = re.sub(r'^Return your final response within \\\\?boxed\{\}\.?(\s*|\n*)', '', question_text)

    instruction = (
            "Return your final response within \\boxed{}.\n"
            "### Question:\n"
            f"{question_text}"
    ).strip()

    # === Output (Accumulate History) ===
    output = ""
    step = 1
    
    for hist in histories[0]:
        tool_calls = hist.get("tool_calls", [])
        active_tool_call = None
        for tool_call in tool_calls:
            if tool_call.get("status", "error") == "success":
                active_tool_call = tool_call
                break
        if not active_tool_call: continue

        call_name = active_tool_call.get("tool_name")
        args_dict = active_tool_call.get("arguments", {}) or {}
        
        reason_text = ""
        final_answer_processed = False
        final_answer_text = ""

        if call_name == "commit":
            reason_text = clean_dsml_tokens(args_dict.get("detailed_reasoning", "").strip())
        
        elif call_name == "final_answer":
            raw_final = clean_dsml_tokens(args_dict.get("final_answer", "").strip())
            
            # Replace Boxed Logic
            final_answer_text = replace_boxed(raw_final, str(model_answers[0]).strip())
            if gt and re.match(r'^[A-Za-z]$', str(gt).strip()):
                final_answer_text = replace_boxed(final_answer_text, str(gt).strip())
                gt_replaced_count += 1
            
            reason_text = args_dict.get("detailed_reasoning", "").strip()
            final_answer_processed = True

        # Append to output string
        if reason_text:
            output += f"[Step {step}] {reason_text}\n\n"
            step += 1
        
        if final_answer_processed:
            # DEFAULT_CHOICES = [
            #     "So now I can conclude the final answer.",
            #     "Now I can provide the final answer.",
            #     "In conclusion, here is the final answer.",
            #     "Therefore, the final answer is as follows.",
            # ]
            # choice = random.choice(DEFAULT_CHOICES)    
            # output += f"{choice}\n\n<answer>{final_answer_text}</answer>\n"
            
            output += f"[Step {step}] {final_answer_text}\n\n"
            step += 1
            
            # extract the boxed answer
            answer = extract_last_boxed(final_answer_text)
            if answer:
                if add_think_tag:
                    output = f"<think>\n{output.strip()}\n</think>\n\n<answer>The final answer is: " + "\\boxed{" + answer + "}</answer>"
                else:
                    output += f"<answer>The final answer is: " + "\\boxed{" + answer + "}</answer>\n"
            

    sample = {
        "id": f"id_{base_id}",
        "instruction": instruction,        
        "output": output.strip()
    }
    
    return [sample], gt_replaced_count


# ==========================================
# Main Process
# ==========================================

def process_data(input_path, output_path, original_file, mode, filtering, add_think_tag=False, strict=False):    
    ids = set()
    if original_file and os.path.exists(original_file):
        for obj in load_jsonl(original_file):
            ids.add(obj.get("id"))
        print(f"Loaded {len(ids)} unique ids from original file.")
    else:
        print(f"Warning: Original file not found at {original_file}. Skipping ID check.")
    
    remain_ids = ids.copy() 
    valid_id_check_set = ids.copy() if ids else None
    duplicate_or_miss_ids = set()
    
    all_examples = []
    
    # Json
    if os.path.isdir(input_path):
        import glob
        for file in glob.glob(os.path.join(input_path, "*.json")):
            new_data = load_json(file)
            if new_data and "examples" in new_data:
                all_examples.extend(new_data["examples"])
        print(f"Loaded {len(all_examples)} examples from directory {input_path}.")
    else:
        data = load_json(input_path)
        if data:
            all_examples = data.get("examples", [])
            print(f"Loaded {len(all_examples)} examples from file {input_path}.")
        
    # Jsonl
    # if os.path.isdir(input_path):
    
    if os.path.isdir(input_path):
        import glob
        for file in glob.glob(os.path.join(input_path, "*.jsonl")):
            all_examples.extend(list(load_jsonl(file)))
        print(f"Loaded {len(all_examples)} examples from directory {input_path}.")
        for file in glob.glob(os.path.join(input_path, "*.jsonl")):
            print(f"File {file} has {len(list(load_jsonl(file)))} examples.")
    else:
        all_examples = list(load_jsonl(input_path))
        print(f"Loaded {len(all_examples)} examples from file {input_path}.")

    if not all_examples:
        print("No examples to process.")
        return

    # 3. Validation Phase
    print(f"=== Starting Validation (Mode: {mode}, Filtering: {filtering}) ===")
    valid_trajectories = []
    abort_samples_ids = {} 
    constraint_counts = defaultdict(int)

    for obj in all_examples:
        base_id = obj.get("id", "unknown_id")
        
        is_valid, reason, stats = validate_trajectory(obj, valid_id_check_set, filtering=filtering)
        
        if is_valid:
            valid_trajectories.append(obj)
            if base_id in remain_ids:
                remain_ids.remove(base_id)
        else:
            if reason == "duplicate or unknown id":
                duplicate_or_miss_ids.add(base_id)
            else:
                abort_samples_ids[base_id] = reason                
            
            for k, v in stats.items():
                constraint_counts[k] += v

    print(f"Validation complete. Valid trajectories: {len(valid_trajectories)}")

    # 4. Conversion Phase
    print("=== Starting Conversion Phase ===")
    final_samples = []
    total_gt_replaced = 0

    if mode != "both":
        convert_func = convert_stepwise if mode == 'stepwise' else convert_full_process

        for obj in valid_trajectories:
            samples, gt_rep = convert_func(obj, filtering=filtering, add_think_tag=add_think_tag)
            final_samples.extend(samples)
            total_gt_replaced += gt_rep

        # 5. Output
        os.makedirs(output_path, exist_ok=True)
        res_file = os.path.join(output_path, f"llama_factory_{mode}_total{len(final_samples)}.json")
        
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(final_samples, f, ensure_ascii=False, indent=2)

        # Logs
        with open(os.path.join(output_path, "aborted_samples.log"), "w", encoding="utf-8") as f:
            for sid, reason in abort_samples_ids.items():
                f.write(f"{sid}\t{reason}\n")
        
        with open(os.path.join(output_path, "duplicate_or_miss_ids.log"), "w", encoding="utf-8") as f:
            for sid in duplicate_or_miss_ids:
                f.write(f"{sid}\n")

        with open(os.path.join(output_path, "remaining_ids.log"), "w", encoding="utf-8") as f:
            for sid in remain_ids:
                f.write(f"{sid}\n")

        print(f"=== Sample preview (Mode: {mode}) ===")
        for i, s in enumerate(final_samples[:2]):
            print(f"\n--- Sample {i} ---")
            print("id:", s["id"])
            print("instruction:\n", s["instruction"][:200] + "..." if len(s["instruction"])>200 else s["instruction"])
            print("output:\n", s["output"][:200] + "..." if len(s["output"])>200 else s["output"])

        print(f"\n✅ 。: {len(final_samples)}。GT: {total_gt_replaced}")
        print(f": {res_file}")
        print(f":")
        print(f"  - Aborted: {len(abort_samples_ids)}")
        print(f"  - Unprocessed IDs: {len(remain_ids)}")
        print(f"  - Constraints: {dict(constraint_counts)}")        
                
    else:
        # output both versions in two separate files
        # make sure ids in stepwise is same as full version
        stepwise_samples = []
        full_samples = []
        _valid_ids: set = set()
        for obj in valid_trajectories:
            step_samples, gt_rep_step = convert_stepwise(obj, filtering=filtering, add_think_tag=add_think_tag, strict=strict, _valid_ids=_valid_ids)
            full_samples_part, gt_rep_full = convert_full_process(obj, filtering=filtering, add_think_tag=add_think_tag, _valid_ids=_valid_ids)
            
            if step_samples and full_samples_part:
                stepwise_samples.extend(step_samples)
                full_samples.extend(full_samples_part)
                total_gt_replaced += (gt_rep_step + gt_rep_full)

        # Output both versions
        os.makedirs(output_path, exist_ok=True)
        stepwise_file = os.path.join(output_path, f"llama_factory_stepwise_total{len(stepwise_samples)}.json")
        full_file = os.path.join(output_path, f"llama_factory_full_total{len(full_samples)}.json")

        with open(stepwise_file, "w", encoding="utf-8") as f:
            json.dump(stepwise_samples, f, ensure_ascii=False, indent=2)
        
        with open(full_file, "w", encoding="utf-8") as f:
            json.dump(full_samples, f, ensure_ascii=False, indent=2)

        print(f"=== Sample preview (Stepwise) ===")
        for i, s in enumerate(stepwise_samples[:2]):
            print(f"\n--- Sample {i} ---")
            print("id:", s["id"])
            print("instruction:\n", s["instruction"][:200] + "..." if len(s["instruction"])>200 else s["instruction"])
            print("output:\n", s["output"][:200] + "..." if len(s["output"])>200 else s["output"])
        print(f"\n=== Sample preview (Full) ===")
        for i, s in enumerate(full_samples[:2]):
            print(f"\n--- Sample {i} ---")
            print("id:", s["id"])
            print("instruction:\n", s["instruction"][:200] + "..." if len(s["instruction"])>200 else s["instruction"])
            print("output:\n", s["output"][:200] + "..." if len(s["output"])>200 else s["output"])

        print(f"\n✅ 。: Stepwise={len(stepwise_samples)}, Full={len(full_samples)}。GT: {total_gt_replaced}")
        print(f": {stepwise_file}, {full_file}")
        print(f":")
        print(f"  - Aborted: {len(abort_samples_ids)}")
        print(f"  - Unprocessed IDs: {len(remain_ids)}")
        print(f"  - Constraints: {dict(constraint_counts)}")    
        print(f"  - Duplicate/Missing IDs: {len(duplicate_or_miss_ids)}")    
        print(f"  - Global valid IDs count: {len(_valid_ids)}")
    
    return remaining_ids_file if 'remaining_ids_file' in locals() else os.path.join(output_path, "remaining_ids.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" LLaMA Factory ")
    parser.add_argument("--original_file", type=str, help=" JSONL  (ID)", default=None)
    parser.add_argument("--input", type=str, required=True, help=" JSON/JSONL ")
    parser.add_argument("--output", type=str, required=True, help="")
    parser.add_argument("--filtering", action="store_true", help="")
    parser.add_argument("--mode", type=str, choices=['stepwise', 'full', "both"], default='both', 
                        help=": 'stepwise' (, Script 1)  'full' ( CoT, Script 2)")
    parser.add_argument("--add_think_tag", action="store_true", help=" commit, expand, fold  reasoning  <tool_call> ")
    parser.add_argument("--strict", action="store_true", help="/ reasoningsummary ()")

    args = parser.parse_args()
    
    rem_file = process_data(args.input, args.output, args.original_file, args.mode, args.filtering, add_think_tag=args.add_think_tag, strict=args.strict)

    if args.original_file and os.path.exists(args.original_file):
        remain_jsonl_output = os.path.join(args.output, "remaining_source_data.jsonl")
        extract_remaining_samples(args.original_file, rem_file, remain_jsonl_output)