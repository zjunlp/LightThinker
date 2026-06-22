import json
import json5
import re
import random
import os
import multiprocessing
from functools import partial
from typing import List, Dict, Any, Optional, Tuple
import logging
import datetime
from tqdm import tqdm  # pip install tqdm

try:
    import orjson
    def fast_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
except ImportError:
    import json as orjson
    def fast_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = """You are a deep research assistant equipped with a Hierarchical Memory System. Your core function is to conduct thorough, multi-source investigations while efficiently managing your context window.

# Memory ID System
You must track your reasoning and observations using specific ID tags to manage context density:
1. **Raw Level** (Default): [Thought ID: N], [Observation ID: N]
2. **Folded Level** (Compressed): [Folded Thought ID: N], [Folded Observation ID: N]
3. **Expanded Level** (Re-activated): [Expanded Thought ID: N], [Expanded Observation ID: N]

# Operation Principles
- **Compression**: Use `commit` to summarize Raw Thought/Observation.
- **Retrieval**: Use `expand` to restore details from a Folded step.
- **Final Answer**: Wrap your response within <answer></answer> tags.

# Tools
You may call one or more functions to assist with the user query.
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}}, "goal": {"type": "string"}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "expand", "description": "Expand the details of a specific Thought or Observation in the memory so you can refer to a more detailed version in your next reasoning step.", "parameters": {"type": "object", "properties": {"thought_id": {"type": "integer"}, "obs_id": {"type": "integer"}, "reason": {"type": "string"}}, "required": []}}}
{"type": "function", "function": {"name": "fold", "description": "Fold back an expanded Thought or Observation into its concise summary form to reduce context length.", "parameters": {"type": "object", "properties": {"thought_id": {"type": "integer"}, "obs_id": {"type": "integer"}}, "required": []}}}
{"type": "function", "function": {"name": "commit", "description": "Commit the current observation into the hierarchical memory by writing a concise summary.", "parameters": {"type": "object", "properties": {"reasoning": {"type": "string"}, "observation_summary": {"type": "string"}}, "required": ["reasoning", "observation_summary"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


TEMPLATE_PATTERN = re.compile(r"\s+(?:Thought|Observation) ID \d+\s+has been expanded")
JSON_CLEAN_PATTERN = re.compile(r"^\s*\{.*?\}(?:\n|\s)*", re.DOTALL)
TOOLRESP_JSON_CLEAN_PATTERN = re.compile(r"(<tool_response>\s*)\{.*?\}(?:\n|\s)*", re.DOTALL)
ID_PATTERN = re.compile(
    r'^\s*\[(Thought|Folded Thought|Expanded Thought|Observation|Folded Observation|Expanded Observation) ID: \d+\]\s*'
)
INNER_RESPONSE_PATTERN = re.compile(r'^\s*\[(Thought|Expanded Thought|Observation|Expanded Observation) ID: \d+\]\s*')

OBS_TAG_PATTERN = re.compile(r'\[Observation ID: (\d+)\]')
THOUGHT_TAG_PATTERN = re.compile(r'\[Thought ID: (\d+)\]')

THOUGHT_ANY_PATTERN = re.compile(r"\[(?:Thought|Expanded Thought|Folded Thought) ID: (\d+)\]")
OBS_ANY_PATTERN = re.compile(r"\[(?:Observation|Expanded Observation|Folded Observation) ID: (\d+)\]")


COMMIT_THOUGHT_TEMPLATES = [
    "I have gathered the necessary information from this step. I will commit it to memory to keep the context concise.",
    "The search results verify the current point. I should summarize this observation and commit it.",
    "This step is complete. Committing the findings to the hierarchical memory now.",
    "To avoid context overflow, I will compress this reasoning and observation into a summary.",
    "Relevant details found. I'll save this to memory and proceed to the next step.",
    "I have analyzed the search results. Writing a concise summary to memory.",
    "Observation is sufficient. Executing commit to free up context window.",
    "I'll document these findings in the memory system before moving on.",
    "Key information extracted. I am committing this to the long-term memory.",
    "The verification for this sub-task is done. I will fold this into a summary.",
    "I have obtained the required data. Saving the current state to memory.",
    "To maintain a clear reasoning path, I will commit this observation.",
    "The data collected is relevant. I will commit it now.",
]

def get_random_commit_thought() -> str:
    return random.choice(COMMIT_THOUGHT_TEMPLATES)

def get_source_key(item: Dict[str, Any]) -> str:
    return str(
        item.get("source")
        or item.get("source_id")
        or item.get("dataset")
        or item.get("origin")
        or item.get("task")
        or item.get("task_name")
        or "unknown"
    )

def make_record_key(obj: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    rid = obj.get("id")
    if rid is None:
        return None
    return (str(rid), get_source_key(obj))

def get_tool_calls_v2(rd: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    - “” rd["tool_calls"] （logging ：tool_name/status/arguments）
    -  OpenAI-ish: rd["output"]["raw_response"]["tool_calls"]
    """
    if isinstance(rd.get("tool_calls"), list) and rd["tool_calls"]:
        return rd["tool_calls"]
    return (rd.get("output") or {}).get("raw_response", {}).get("tool_calls") or []

def postprocess_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    if "id" in sample:
        sample["id"] = str(sample["id"])

    sid = (sample.get("id") or "").strip()
    messages = sample.get("messages", [])

    for msg in messages:
        c = msg.get("content", "") or ""
        if "<tool_response>" in c:
            msg["content"] = TOOLRESP_JSON_CLEAN_PATTERN.sub(r"\1", c).strip()

    # if sid.isdigit():
    #     return sample

    is_commit = ("commit" in sid) or ("_c" in sid)
    is_other_b = any(k in sid for k in ["fold", "expand", "final", "_f", "_t"])

    def protected(i: int) -> bool:
        cur = messages[i].get("content", "") or ""
        prev = (messages[i - 1].get("content", "") or "") if i > 0 else ""
        return (
            '"name": "fold"' in cur
            or '"name": "expand"' in cur
            or '"name": "fold"' in prev
            or '"name": "expand"' in prev
        )

    if is_commit:
        limit = len(messages) - 3
        for i in range(len(messages)):
            if i < 2 or i >= limit:
                continue
            if protected(i):
                continue
            content = messages[i].get("content", "") or ""
            content = OBS_TAG_PATTERN.sub(r"[Folded Observation ID: \1]", content)
            messages[i]["content"] = content

    elif is_other_b:
        for i in range(len(messages)):
            if i < 2:
                continue
            if protected(i):
                continue
            content = messages[i].get("content", "") or ""
            content = OBS_TAG_PATTERN.sub(r"[Folded Observation ID: \1]", content)
            messages[i]["content"] = content

        if messages and messages[-1].get("role") == "assistant":
            last_content = messages[-1].get("content", "") or ""
            if "[Folded Thought ID:" in last_content:
                messages[-1]["content"] = last_content.replace("[Folded Thought ID:", "[Thought ID:")

    return sample

def is_simple_trajectory(history: List[Dict[str, Any]]) -> bool:
    memory_tools = {"commit", "expand", "fold"}
    for rd in history:
        for tc in get_tool_calls_v2(rd):
            func = tc.get("function")
            name = func.get("name") if isinstance(func, dict) else (tc.get("tool_name") or tc.get("name"))
            if name in memory_tools:
                return False
        if extract_logger_commits(rd):
            return False

    return True



def validate_memory_logic(history, record_id="unknown"):
    action_sequence = []

    def _parse_args(args_any):
        if args_any is None: return {}
        if isinstance(args_any, dict): return args_any
        if isinstance(args_any, str):
            try: return json5.loads(args_any)
            except Exception: return {}
        return {}

    for rd in history:
        tool_calls = get_tool_calls_v2(rd) + extract_logger_commits(rd)

        for tc in tool_calls:
            func = tc.get("function")
            if isinstance(func, dict):
                name = func.get("name")
                args = _parse_args(func.get("arguments"))
            else:
                name = tc.get("tool_name") or tc.get("name")
                args = _parse_args(tc.get("arguments"))

            if name not in {"commit", "expand", "fold"}:
                continue

            if name in {"expand", "fold"}:
                target = args.get("thought_id")
                if target is None:
                    target = args.get("obs_id")
                target_id = str(target) if target is not None else None
            else:
                target = tc.get("obs_id") or tc.get("thought_id")
                target_id = str(target) if target is not None else None
               
            action_sequence.append({"name": name, "target_id": target_id})

    if not action_sequence:
        return False, "no_memory_actions"


    expanded_ids = set()
    has_expand = False
    has_fold = False
    has_commit = False

    
    for i in range(len(action_sequence)):
        curr = action_sequence[i]
        prev = action_sequence[i-1] if i > 0 else None
        
        if prev and curr["name"] == prev["name"] and curr["target_id"] == prev["target_id"] and curr["target_id"] is not None:
            return False, "consecutive_duplicate_id"
        
        # if prev and prev["name"] == "commit" and curr["name"] == "expand":
        #     if prev["target_id"] is not None and curr["target_id"] is not None:
        #         if curr["target_id"] == prev["target_id"]:
        #             return False, "commit_then_expand"

        
        if curr["name"] == "commit":
            has_commit = True

        if curr["name"] == "expand":
            has_expand = True
            if curr["target_id"]:
                expanded_ids.add(curr["target_id"])
                
        if curr["name"] == "fold":
            has_fold = True
            if not curr["target_id"] or curr["target_id"] not in expanded_ids:
                # print(f"\n[DROPPED_FOLD_ERROR] Record ID: {record_id}")
                # print(f"  Attempted to FOLD ID: {curr['target_id']}")
                # print(f"  Currently Expanded IDs: {expanded_ids}")
                # print(f"  Action Index: {i}")
                # print(f"  Sequence: {[(a['name'], a['target_id']) for a in action_sequence]}")
                return False, "fold_without_previous_expand"
            
            expanded_ids.remove(curr["target_id"])

    if not (has_expand and has_fold and has_commit):
        return False, "missing_ef_mix"
        
    return True, "ok"


def convert_input_messages_to_training_msgs(question: str, input_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], int]:
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": SYSTEM_PROMPT + today_date()})

    cycle_count = 1
    pending_tool_calls_map: Dict[str, str] = {}
    
    for m in input_messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        tool_calls = m.get("tool_calls") or []

        if not content and not tool_calls: continue
        if role == "system": continue

        if role == "user":
            if "<tool_response>" in content:
                if "[Observation ID:" not in content and "[Expanded Observation ID:" not in content:
                    content = content.replace("<tool_response>", f"<tool_response>\n[Observation ID: {cycle_count}]\n", 1)
                msgs.append({"role": "user", "content": content})
                cycle_count += 1
            else:
                msgs.append({"role": "user", "content": content})

        elif role == "assistant":
            parts = []
            if content:
                if not ID_PATTERN.match(content):
                    content = f"[Thought ID: {cycle_count}]\n{content}"
                parts.append(f"<think>\n{content.strip()}\n</think>")
            
            for tc in tool_calls:
                func = tc.get("function") or {}
                name = func.get("name")
                call_id = tc.get("id")
                args = func.get("arguments") or "{}"
                
                if call_id: pending_tool_calls_map[call_id] = name
                
                tc_json = fast_dumps({"name": name, "arguments": json5.loads(args) if isinstance(args, str) else args})
                if name != "commit":
                    parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")

            if parts:
                msgs.append({"role": "assistant", "content": "\n\n".join(parts)})

        elif role == "tool":
            call_id = m.get("tool_call_id")
            tool_name = pending_tool_calls_map.get(call_id, "unknown_tool")

            if tool_name in {"expand", "fold"}:
                match_template = TEMPLATE_PATTERN.search(content)
                if match_template:
                    content = content[match_template.start():].strip()

            if not ID_PATTERN.match(content):
                content = f"[Observation ID: {cycle_count}]\n{content}"
            else:
                content = ID_PATTERN.sub(lambda mo: f"{mo.group(0).strip()}\n", content, count=1)

            msgs.append({"role": "user", "content": f"<tool_response>\n{content.strip()}\n</tool_response>"})
            cycle_count += 1
            
    return msgs, cycle_count

def build_final_answer_message(full_assistant_content: str, thought_id: int) -> Optional[Dict[str, str]]:
    raw = full_assistant_content.strip()
    if "<answer>" not in raw: return None
    idx = raw.index("<answer>")
    think_part, answer_part = raw[:idx].strip(), raw[idx:].strip()
    parts = []
    if think_part:
        id_tag = f"[Thought ID: {thought_id}]"
        content_with_id = ID_PATTERN.sub(f"{id_tag}\n", think_part, 1) if ID_PATTERN.match(think_part) else f"{id_tag}\n{think_part}"
        parts.append(f"<think>\n{content_with_id.strip()}\n</think>")
    parts.append(answer_part)
    return {"role": "assistant", "content": "\n\n".join(parts).strip()}

def build_tool_action_message(full_assistant_content: str, tool_call: Dict[str, Any], allowed: set, thought_id: int) -> Optional[Dict[str, str]]:
    name = tool_call.get("function", {}).get("name")
    if name not in allowed: return None
    args = tool_call.get("function", {}).get("arguments") or {}
    try: arguments = json5.loads(args) if isinstance(args, str) else args
    except: arguments = {}
    tc_str = fast_dumps({"name": name, "arguments": arguments})
    parts = []
    if full_assistant_content:
        id_tag = f"[Thought ID: {thought_id}]"
        content_id = ID_PATTERN.sub(f"{id_tag}\n", full_assistant_content, 1) if ID_PATTERN.match(full_assistant_content) else f"{id_tag}\n{full_assistant_content}"
        parts.append(f"<think>\n{content_id.strip()}\n</think>")
    parts.append(f"<tool_call>\n{tc_str}\n</tool_call>")
    return {"role": "assistant", "content": "\n\n".join(parts).strip()}

def build_commit_action_message(reasoning: str, observation_summary: str, thought_id: int) -> Dict[str, str]:
    tc_str = fast_dumps({"name": "commit", "arguments": {"reasoning": reasoning, "observation_summary": observation_summary}})
    return {"role": "assistant", "content": f"<think>\n[Thought ID: {thought_id}]\n{get_random_commit_thought()}\n</think>\n\n<tool_call>\n{tc_str}\n</tool_call>"}

def extract_raw_thought_and_obs_from_logger_input(input_messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    user_msg = next((m for m in reversed(input_messages) if m.get("role") == "user"), None)
    if not user_msg: return None, None
    text = user_msg.get("content", "")
    idx_t, idx_o = text.find("## Raw Thought:"), text.find("## Raw Observation:")
    if idx_t != -1 and idx_o != -1 and idx_o > idx_t:
        thought = text[idx_t + 15:idx_o].strip() or None
        obs = text[idx_o + 19:].split("----- End of Record -----")[0].strip()
        return thought, (obs if obs else None)
    return None, None

def extract_logger_commits(rd):
    notes = rd.get("notes") or []
    is_logging = any(isinstance(n, str) and "Logging" in n for n in notes)
    if not is_logging:
        return []

    out = []
    for tc in (rd.get("tool_calls") or []):
        tool_name = tc.get("tool_name") or ""
        if not tool_name.startswith("log_step"):
            continue
        if tc.get("status") != "success":
            continue
        args = tc.get("arguments") or {}
        if not (args.get("reasoning") and args.get("observation_summary")):
            continue

        out.append({
            "tool_name": "commit",
            "arguments": {
                "reasoning": args["reasoning"],
                "observation_summary": args["observation_summary"],
            },
            "obs_id": tc.get("obs_id"),
            "thought_id": tc.get("thought_id"),
        })
    return out


def find_prev_main_round(history: List[Dict[str, Any]], idx_logging: int) -> Optional[Dict[str, Any]]:
    for i in range(idx_logging - 1, -1, -1):
        if any("Thinking/Action" in n for n in history[i].get("notes", [])): return history[i]
    return None


def process_batch(batch_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    local_samples = []
    stats = {
        "total": 0,
        "clean_memory_kept": 0,
        "clean_simple_kept": 0,
        "dropped_simple_short": 0,
        "dropped_simple_sampled": 0,
        "dropped_no_ef_mix": 0,
        "dropped_jitter": 0,
        "dropped_fold_error": 0,
        "dropped_too_long": 0,
        "skipped_no_data": 0
    }
    
    MAX_ROUNDS_THRESHOLD = 60
    SIMPLE_SAMPLE_RATE = 0.2
    MIN_SIMPLE_ROUNDS = 3

    for obj in batch_records:
        stats["total"] += 1
        history = obj.get("interaction_history") or []
        if not history: 
            stats["skipped_no_data"] += 1
            continue

        n_rounds = len(history)
        is_simple = is_simple_trajectory(history)

        if is_simple:
            if n_rounds < MIN_SIMPLE_ROUNDS:
                stats["dropped_simple_short"] += 1
                continue
            # if random.random() > SIMPLE_SAMPLE_RATE:
            #     stats["dropped_simple_sampled"] += 1
            #     continue
            stats["clean_simple_kept"] += 1
        else:
            if n_rounds > MAX_ROUNDS_THRESHOLD:
                stats["dropped_too_long"] += 1
                continue
            
            is_valid, reason = validate_memory_logic(history, obj.get('id'))
            if not is_valid:
                if reason == "missing_ef_mix": stats["dropped_no_ef_mix"] += 1
                elif reason in ["consecutive_duplicate_id", "commit_then_expand"]: stats["dropped_jitter"] += 1
                else: stats["dropped_fold_error"] += 1
                continue
            stats["clean_memory_kept"] += 1
        
        base_id = obj.get("id", "unk")
        for idx, rd in enumerate(history):
            notes = rd.get("notes") or []
            is_main = any("Thinking/Action" in n for n in notes)
            is_logging = any("Logging" in n for n in notes)
            round_idx = rd.get("round")

            if is_main:
                assistant_content = str((rd.get("output") or {}).get("raw_response", {}).get("content", "")).strip()
                tool_calls = (rd.get("output") or {}).get("raw_response", {}).get("tool_calls") or []
                ctx_msgs, next_id = convert_input_messages_to_training_msgs(obj.get("question", ""), rd.get("input_messages", []))
                
                final_msg = build_final_answer_message(assistant_content, next_id)
                if final_msg: 
                    local_samples.append({"id": f"{base_id}_r{round_idx}_final", "messages": ctx_msgs + [final_msg]})

                for ti, tc in enumerate(tool_calls):
                    if tc.get("function", {}).get("name") in {"expand", "fold"}:
                        name = tc.get("function", {}).get("name")
                        act_msg = build_tool_action_message(assistant_content, tc, {"expand", "fold"}, next_id)
                        if act_msg: local_samples.append({"id": f"{base_id}_r{round_idx}_{name}_{ti}", "messages": ctx_msgs + [act_msg]})

            if is_logging:
                prev_main = find_prev_main_round(history, idx)
                if not prev_main: continue
                ctx_msgs, next_id = convert_input_messages_to_training_msgs(obj.get("question", ""), prev_main.get("input_messages", []))
                _, raw_obs = extract_raw_thought_and_obs_from_logger_input(rd.get("input_messages", []))
                if not raw_obs: continue
                
                prev_out = (prev_main.get("output") or {}).get("raw_response", {}).get("content", "") or "Okay."
                clean_assist = re.sub(r'\[(Thought|Expanded Thought) ID: \d+\]\s*', '', str(prev_out)).strip()
                
                success_calls = [tc for tc in prev_main.get("tool_calls", []) if tc.get("status") == "success" and tc.get("tool_name") not in {"expand", "fold", "commit"}]
                if not success_calls: continue
                
                tc_str = fast_dumps({"name": success_calls[0]["tool_name"], "arguments": success_calls[0].get("arguments", {})})
                base_ctx = ctx_msgs + [{"role": "assistant", "content": f"<think>\n[Thought ID: {next_id}]\n{clean_assist}\n</think>\n\n<tool_call>\n{tc_str}\n</tool_call>"}]
                tr_block = f"<tool_response>\n[Observation ID: {next_id}]\n{raw_obs}\n</tool_response>"
                ctx_before_commit = base_ctx + [{"role": "user", "content": tr_block}]

                for li, tc in enumerate(rd.get("tool_calls") or []):
                    if (tc.get("tool_name") or "").startswith("log_step") and tc.get("status") == "success":
                        args = tc.get("arguments") or {}
                        if args.get("reasoning") and args.get("observation_summary"):
                            commit_msg = build_commit_action_message(args["reasoning"], args["observation_summary"], next_id + 1)
                            local_samples.append({"id": f"{base_id}_r{round_idx}_commit_{li}", "messages": ctx_before_commit + [commit_msg]})

    return local_samples, stats


def calculate_sample_length(sample: Dict[str, Any]) -> int:
    return sum(len(m.get("content", "")) for m in sample.get("messages", []))

def deduplicate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    shortest_map = {}
    for s in samples:
        sid = s.get("id")
        if not sid: continue
        length = calculate_sample_length(s)
        if sid not in shortest_map or length < calculate_sample_length(shortest_map[sid]):
            shortest_map[sid] = s
    return list(shortest_map.values())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter and convert raw DeepResearch trajectories into SFT training data.")
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more input JSONL files containing raw interaction histories.")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path for the curated training data.")
    cli = parser.parse_args()

    BASE_FILES = cli.input
    OUT_FILE = cli.output

    all_records: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, str]] = set()

    for fp in BASE_FILES:
        if not os.path.exists(fp):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                key = make_record_key(obj)   # (id, source)
                if key is None:
                    continue

                if key in seen_keys:
                    continue

                seen_keys.add(key)
                all_records.append(obj)


    num_procs = min(multiprocessing.cpu_count(), 32)
    chunk_size = max(1, len(all_records) // num_procs)
    chunks = [all_records[i:i + chunk_size] for i in range(0, len(all_records), chunk_size)]
    
    all_gen = []
    final_stats = {k: 0 for k in ["total", "clean_memory_kept", "clean_simple_kept", "dropped_simple_short", "dropped_simple_sampled", "dropped_no_ef_mix", "dropped_jitter", "dropped_fold_error", "dropped_too_long", "skipped_no_data"]}

    with multiprocessing.Pool(processes=num_procs) as pool:
        results = list(tqdm(pool.imap(process_batch, chunks), total=len(chunks)))
        for batch_samples, batch_stats in results:
            all_gen.extend(batch_samples)
            for k, v in batch_stats.items(): final_stats[k] += v

    final_samples = deduplicate_samples(all_gen)
    final_samples = [postprocess_sample(s) for s in final_samples]

    
    print("\n" + "="*60)
    print("DATA CURATION REPORT")
    print("="*60)
    print(f"Total Records:      {final_stats['total']}")
    print(f"✅ Memory Kept:     {final_stats['clean_memory_kept']}")
    print(f"✅ Simple Kept:     {final_stats['clean_simple_kept']}")
    print("-" * 30)
    print(f"🚫 Drop (No EF Mix):   {final_stats['dropped_no_ef_mix']}")
    print(f"🚫 Drop (Fold Error):  {final_stats['dropped_fold_error']}")
    print(f"🚫 Drop (Jitter):      {final_stats['dropped_jitter']}")
    print(f"🚫 Drop (Too Long):    {final_stats['dropped_too_long']}")
    print(f"🚫 Drop (Simp Sample): {final_stats['dropped_simple_sampled']}")
    print("="*60)
    print(f"Final Output Size:  {len(final_samples)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()