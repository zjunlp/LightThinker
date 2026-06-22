#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

DATASETS_DEFAULT =["gsm8k", "mmlu", "gpqa", "bbh"]

DATASET_EXPECTED_COUNTS = {
    "gsm8k": 1319,
    "mmlu": 1027,
    "gpqa": 198,
    "bbh": 495
}

@dataclass
class Metrics:
    acc: Optional[float] = None
    peak: Optional[float] = None
    dep: Optional[float] = None
    file: Optional[str] = None
    num_repeats: int = 1 
    selected_files: Optional[List[str]] = None
    total_time_seconds: Optional[float] = None
    sec_per_sample: Optional[float] = None
    
    avg_turn: Optional[float] = None
    avg_total_generated_tokens: Optional[float] = None

@dataclass
class AvgMetrics:
    avg_acc: Optional[float] = None
    avg_peak: Optional[float] = None
    avg_dep: Optional[float] = None
    avg_sec_per_sample: Optional[float] = None 
    avg_total_time: Optional[float] = None 
    avg_turn: Optional[float] = None
    avg_total_generated_tokens: Optional[float] = None


def _digit_compare(model_ans: str, gt_ans: str):
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

def _find_first_list_by_keys(obj, keys=("examples", "data")):
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k, None)
            if isinstance(v, list):
                return v
        for v in obj.values():
            r = _find_first_list_by_keys(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_first_list_by_keys(item, keys)
            if r is not None:
                return r
    return None

def _safe_get_avg_sec_per_sample(data: Any) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    candidates =[
        data.get("avg_sec_per_sample"),
        (data.get("results") or {}).get("avg_sec_per_sample") if isinstance(data.get("results"), dict) else None,
        (data.get("meta") or {}).get("avg_sec_per_sample") if isinstance(data.get("meta"), dict) else None,
    ]
    for v in candidates:
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def _safe_get_total_time_seconds_wall(data: Any) -> Optional[float]:
    if not isinstance(data, dict):
        return None
    candidates =[
        data.get("total_time_seconds_wall"),
        data.get("total_time_seconds"),
        (data.get("meta") or {}).get("total_time_seconds_wall") if isinstance(data.get("meta"), dict) else None,
    ]
    for v in candidates:
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def load_examples_and_time(file_path: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[float], Optional[float]]:
    try:        
        if file_path.lower().endswith(".jsonl"):
            data =[]
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                    except Exception:
                        continue
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception:
        return None, None, None

    avg_sps = _safe_get_avg_sec_per_sample(data)
    wall = _safe_get_total_time_seconds_wall(data)

    if isinstance(data, list):
        return data, avg_sps, wall

    if isinstance(data, dict):
        exs = _find_first_list_by_keys(data, keys=("examples", "data"))
        exs = exs if isinstance(exs, list) else[]
        return exs, avg_sps, wall

    return[], avg_sps, wall


def calc_accuracy_and_errors(examples: List[Dict[str, Any]], dataset_name: str) -> Tuple[Optional[float], int, List[Dict[str, Any]]]:
    if not examples:
        return None, 0,[]

    total_count = 0
    correct_count = 0
    wrong_cases =[]

    for ex in examples:
        gold = str(ex.get("answer", "") or ex.get("ground_truth", "")).strip().lower()
        pred = str(ex.get("model_answer", "") or ex.get("prediction", "")).strip().lower()
        question = ex.get("question") or ex.get("input") or ex.get("query") or "N/A"

        # in case there are wrapping \\text{}
        gold=re.sub(r"\\text\{([^}]*)\}", r"\1", gold)
        pred=re.sub(r"\\text\{([^}]*)\}", r"\1", pred)

        if gold == "":
            continue

        is_correct = False

        if gold == pred:
            is_correct = True
        elif dataset_name in ["gpqa", "mmlu"]:
            pattern = r"^([A-E])[\).]?"
            gm = re.match(pattern, gold, re.IGNORECASE)
            pm = re.match(pattern, pred, re.IGNORECASE)
            if gm and pm and gm.group(1).upper() == pm.group(1).upper():
                is_correct = True
        elif dataset_name == "gsm8k":
            if _digit_compare(pred, gold) is True:
                is_correct = True
        elif dataset_name == "bbh":
            matched = False
            pattern = r"^([A-E])[\).]?"
            gm = re.match(pattern, gold, re.IGNORECASE)
            pm = re.match(pattern, pred, re.IGNORECASE)
            if gm and pm and gm.group(1).upper() == pm.group(1).upper():
                is_correct = True
                matched = True
            if not matched:
                if _digit_compare(pred, gold) is True:
                    is_correct = True

        if is_correct and pred != "error":
            correct_count += 1
        else:
            wrong_cases.append({
                "question": question,
                "gold": gold,
                "pred": pred,
            })

        total_count += 1

    acc = (100.0 * correct_count / total_count) if total_count > 0 else None
    return acc, total_count, wrong_cases

def calc_avg_turn_and_total_generated_tokens(examples: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not examples:
        return None, None
    
    total_turns = 0
    total_genertaed_tokens = 0
    count = 0
    
    for ex in examples:
        hist = ex.get("model_full_history", []) or ex.get("history", []) or[]
        turns = len(hist)
        total_tokens = ex.get("total_output_tokens", 0) or 0
        if turns > 0:
            total_turns += turns
            total_genertaed_tokens += total_tokens
            count += 1

    avg_turns = (total_turns / len(examples)) if count > 0 else None
    avg_total_generated_tokens = (total_genertaed_tokens / len(examples)) if count > 0 else None
    return avg_turns, avg_total_generated_tokens

def calc_peak_tokens(examples: List[Dict[str, Any]]) -> Optional[float]:
    if not examples:
        return None
    
    total_max_tokens = 0
    found_any_round = False
    
    for ex in examples:
        hist = ex.get("model_full_history", []) or ex.get("history", []) or[]
        peak_tokens = 0
        for rd in hist:
            inp = rd.get("input_tokens", 0) or 0
            out = rd.get("output_tokens", 0) or 0
            total = inp + out
            found_any_round = True
            if total > peak_tokens:
                peak_tokens = total
        total_max_tokens += peak_tokens

    return (total_max_tokens / len(examples)) if found_any_round and len(examples) > 0 else None

def calc_dependency(examples: List[Dict[str, Any]], mode: str) -> Optional[float]:
    if not examples:
        return None

    total_dep = 0.0
    found_any = False

    if mode == "vanilla":
        for ex in examples:
            hist = ex.get("model_full_history", []) or ex.get("history", []) or[]
            dep = 0.0
            for rd in hist:
                inp = rd.get("input_tokens", 0) or 0
                out = rd.get("output_tokens", 0) or 0
                dep = (inp + inp + out) * out * 0.5
                found_any = True
            total_dep += dep
    elif mode == "ltthinker":
        for ex in examples:
            hist = ex.get("model_full_history", []) or ex.get("history", []) or[]
            dep = 0.0
            for rd in hist:
                inp = rd.get("input_tokens", 0) or 0
                out = rd.get("output_tokens", 0) or 0
                dep += (inp + inp + out) * out * 0.5
                found_any = True
            total_dep += dep
    else:
        raise ValueError(f"Unknown dependency mode: {mode}")

    return (total_dep / len(examples)) if found_any and len(examples) > 0 else None

def iter_json_files(root_dir: str):
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".json") and "results" in fn:
                yield os.path.join(r, fn)

def iter_json_or_jsonl_files(root_dir: str):
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if (fn.lower().endswith(".json") or fn.lower().endswith(".jsonl")) and "results" in fn:
                yield os.path.join(r, fn)

def parse_model_and_dataset(file_path: str, dataset_names: List[str]) -> Optional[Tuple[str, str]]:
    parts = os.path.normpath(file_path).split(os.sep)
    ds_lower_map = {d.lower() for d in dataset_names}

    ds_idx = -1
    dataset = None

    for i, part in enumerate(parts):
        if part.lower() in ds_lower_map:
            ds_idx = i
            dataset = part.lower()
            break
    
    if ds_idx == -1 or (ds_idx - 1) < 0 or (ds_idx + 2) >= len(parts):
        return None

    base_model = parts[ds_idx - 1]
    config_folder = parts[ds_idx + 2] 

    model_id = f"{base_model}__{config_folder}"
        
    return model_id, dataset

def aggregate_repeats(
    raw_results: Dict[str, Dict[str, List[Metrics]]],
    max_repeats: int = 3,
    print_selected: bool = False,
) -> Dict[str, Dict[str, Metrics]]:
    aggregated: Dict[str, Dict[str, Metrics]] = {}

    def dep_key(m: Metrics):
        return -1e18 if m.dep is None else m.dep

    for model, ds_map in raw_results.items():
        aggregated[model] = {}
        for ds, m_list in ds_map.items():
            if not m_list:
                continue

            if len(m_list) <= max_repeats:
                selected = list(m_list)
            else:
                selected = random.sample(m_list, k=max_repeats)

            selected = sorted(selected, key=dep_key, reverse=True)

            if print_selected:
                print(f"\n[SELECT] model={model} dataset={ds} total_files={len(m_list)} selected={len(selected)} max_repeats={max_repeats}")
                for j, m in enumerate(selected, 1):
                    dep_str = "None" if m.dep is None else f"{m.dep/1e6:.2f}M"
                    acc_str = "None" if m.acc is None else f"{m.acc:.2f}"
                    peak_str = "None" if m.peak is None else f"{m.peak:.0f}"
                    time_h_str = "None" if m.total_time_seconds is None else f"{m.total_time_seconds/3600:.2f}h"
                    turn_str = "None" if m.avg_turn is None else f"{m.avg_turn:.2f}"
                    gen_tok_str = "None" if m.avg_total_generated_tokens is None else f"{m.avg_total_generated_tokens:.0f}"
                    print(f"  #{j}: acc={acc_str} peak={peak_str} dep={dep_str} time={time_h_str} turn={turn_str} gen_tok={gen_tok_str} file={m.file}")

            valid_accs = [m.acc for m in selected if m.acc is not None]
            valid_peaks = [m.peak for m in selected if m.peak is not None]
            valid_deps  = [m.dep for m in selected if m.dep is not None]

            valid_sps =[m.sec_per_sample for m in selected if m.sec_per_sample is not None]
            avg_sps = sum(valid_sps) / len(valid_sps) if valid_sps else None

            valid_tts =[m.total_time_seconds for m in selected if m.total_time_seconds is not None]
            avg_tts = sum(valid_tts) / len(valid_tts) if valid_tts else None
            
            valid_turns =[m.avg_turn for m in selected if m.avg_turn is not None]
            avg_turns_agg = sum(valid_turns) / len(valid_turns) if valid_turns else None

            valid_gen_tokens =[m.avg_total_generated_tokens for m in selected if m.avg_total_generated_tokens is not None]
            avg_gen_tokens_agg = sum(valid_gen_tokens) / len(valid_gen_tokens) if valid_gen_tokens else None

            avg_acc  = sum(valid_accs) / len(valid_accs) if valid_accs else None
            avg_peak = sum(valid_peaks) / len(valid_peaks) if valid_peaks else None
            avg_dep  = sum(valid_deps) / len(valid_deps) if valid_deps else None

            first_file = selected[0].file if selected else None

            aggregated[model][ds] = Metrics(
                acc=avg_acc,
                peak=avg_peak,
                dep=avg_dep,
                file=first_file,
                num_repeats=len(selected),
                selected_files=[m.file for m in selected if m.file],
                total_time_seconds=avg_tts,
                sec_per_sample=avg_sps,
                avg_turn=avg_turns_agg,
                avg_total_generated_tokens=avg_gen_tokens_agg,
            )

    return aggregated


def get_average_metrics(model_metrics: Dict[str, Metrics], target_datasets: List[str]) -> AvgMetrics:
    acc_sum, acc_cnt = 0.0, 0
    peak_sum, peak_cnt = 0.0, 0
    dep_sum, dep_cnt = 0.0, 0
    sps_sum, sps_cnt = 0.0, 0
    time_sum, time_cnt = 0.0, 0  
    turn_sum, turn_cnt = 0.0, 0
    gen_tokens_sum, gen_tokens_cnt = 0.0, 0

    for ds in target_datasets:
        m = model_metrics.get(ds)
        if not m:
            continue
        
        if m.acc is not None:
            acc_sum += m.acc
            acc_cnt += 1
        
        if m.peak is not None:
            peak_sum += m.peak
            peak_cnt += 1
            
        if m.dep is not None:
            dep_sum += m.dep
            dep_cnt += 1

        if m.sec_per_sample is not None:
            sps_sum += m.sec_per_sample
            sps_cnt += 1
        
        if m.total_time_seconds is not None:
            time_sum += m.total_time_seconds
            time_cnt += 1
            
        if m.avg_turn is not None:
            turn_sum += m.avg_turn
            turn_cnt += 1
            
        if m.avg_total_generated_tokens is not None:
            gen_tokens_sum += m.avg_total_generated_tokens
            gen_tokens_cnt += 1

    return AvgMetrics(
        avg_acc=(acc_sum / acc_cnt) if acc_cnt > 0 else None,
        avg_peak=(peak_sum / peak_cnt) if peak_cnt > 0 else None,
        avg_dep=(dep_sum / dep_cnt) if dep_cnt > 0 else None,
        avg_sec_per_sample=(sps_sum / sps_cnt) if sps_cnt > 0 else None,
        avg_total_time=(time_sum / time_cnt) if time_cnt > 0 else None,
        avg_turn=(turn_sum / turn_cnt) if turn_cnt > 0 else None,
        avg_total_generated_tokens=(gen_tokens_sum / gen_tokens_cnt) if gen_tokens_cnt > 0 else None
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing the JSON result files to process")
    ap.add_argument("--datasets", default="gsm8k,mmlu,gpqa,bbh", help="list of datasets to include, comma separated, e.g. 'gsm8k,mmlu'")
    ap.add_argument("--dependency_mode", choices=["vanilla", "ltthinker"], default="ltthinker")
    ap.add_argument("--show_files", action="store_true", help="whether to print info of each processed file")
    ap.add_argument("--out_leaderboard_json", default=None, help="output path for the aggregated leaderboard JSON file")
    ap.add_argument("--out_wrong_json", default=None, help="output path for the wrong cases JSON file")    
    ap.add_argument("--max_repeats", type=int, default=3, help="maximum number of files to aggregate per model-dataset pair (default: 3)")
    ap.add_argument("--seed", type=int, default=42)    
    ap.add_argument("--allow_incomplete", action="store_true", help="whether to allow datasets with incomplete number of examples")

    args = ap.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)

    input_dir = args.input_dir
    target_datasets =[x.strip() for x in args.datasets.split(",") if x.strip()]
    target_set = set(target_datasets)

    out_path = args.out_leaderboard_json
    if not out_path:
        out_path = os.path.join(input_dir, "leaderboard.json")
    
    out_wrong_path = args.out_wrong_json
    if not out_wrong_path:
        out_wrong_path = os.path.join(input_dir, "wrong_cases.json")

    raw_results: Dict[str, Dict[str, List[Metrics]]] = defaultdict(lambda: defaultdict(list))
    all_wrong_cases: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    print(f">>> Scanning files in: {input_dir}")
    print(f">>> Enforcing dataset counts: {DATASET_EXPECTED_COUNTS}")
    print(f">>> Max repeats allowed: {args.max_repeats}")
    if args.allow_incomplete:
        print(f">>> [WARNING] Strict count check DISABLED (--allow_incomplete is set)")
    else:
        print(f">>> Strict count check ENABLED (Incomplete files will be skipped)")
    
    files_processed = 0
    # for fp in iter_json_or_jsonl_files(input_dir):
    for fp in iter_json_files(input_dir):
        parsed = parse_model_and_dataset(fp, DATASETS_DEFAULT)
        if not parsed:
            continue
        
        model_id, dataset = parsed
        
        if dataset not in target_set:
            continue

        examples, avg_sps_from_file, wall_time = load_examples_and_time(fp)

        if examples is None:
            if args.show_files:
                print(f"[WARN] Invalid JSON or load error: {fp}")
            continue

        
        actual_count = len(examples)
        expected_count = DATASET_EXPECTED_COUNTS.get(dataset)
        
        sec_per_sample = None
        total_time_seconds = None

        if avg_sps_from_file is not None:
            sec_per_sample = avg_sps_from_file
            total_time_seconds = wall_time

        elif wall_time is not None and actual_count > 0:
            sec_per_sample = wall_time / actual_count
            total_time_seconds = wall_time
    
        if expected_count is not None and actual_count != expected_count:
            if not args.allow_incomplete:
                print(f"[SKIP] Incomplete Dataset: {dataset} | Found: {actual_count} | Expected: {expected_count} | File: {os.path.basename(fp)}")
                continue
            else:
                if args.show_files:
                    print(f"[WARN] Processing incomplete Dataset: {dataset} | Found: {actual_count}/{expected_count} | File: {os.path.basename(fp)}")

        acc, total_cnt, errors = calc_accuracy_and_errors(examples, dataset)
        peak = calc_peak_tokens(examples)
        dep = calc_dependency(examples, mode=args.dependency_mode)
        
        avg_turn, avg_total_generated_tokens = calc_avg_turn_and_total_generated_tokens(examples)
        
        raw_results[model_id][dataset].append(
            Metrics(acc=acc, peak=peak, dep=dep, file=fp,
                    total_time_seconds=total_time_seconds,
                    sec_per_sample=sec_per_sample,
                    avg_turn=avg_turn,
                    avg_total_generated_tokens=avg_total_generated_tokens)
        )
        files_processed += 1

        if errors:
            all_wrong_cases[model_id][dataset].append({
                "file": fp,
                "error_count": len(errors),
                "cases": errors
            })

        if args.show_files:
            print(f"[LOADED] ID={model_id} ds={dataset} cnt={actual_count} acc={acc:.2f} errs={len(errors)} file={os.path.basename(fp)}")

    if files_processed == 0:
        print("[ERROR] No valid result files found! Check input directory, file structure, and data completeness.")
        return

    
    results = aggregate_repeats(raw_results, max_repeats=args.max_repeats)
    
    model_avgs: Dict[str, AvgMetrics] = {}
    for model, d_map in results.items():
        model_avgs[model] = get_average_metrics(d_map, target_datasets)

    def sort_key(model: str):
        avgs = model_avgs.get(model, AvgMetrics())
        primary = avgs.avg_acc if avgs.avg_acc is not None else -1.0
        vec =[]
        for ds in DATASETS_DEFAULT:
            if ds in target_set:
                v = results.get(model, {}).get(ds, Metrics()).acc
                vec.append(-1.0 if v is None else v)
        return (primary, tuple(vec))

    models_sorted = sorted(results.keys(), key=sort_key, reverse=True)

    print(f"\n=== Summary (Merged Repeats [Max={args.max_repeats}] & Sorted by Avg Acc) ===")
    
    
    header =["Model_Config", "AVG_Acc", "AVG_Time(h)", "AVG_Sec/Samp", "AVG_Peak", "AVG_Dep(M)", "AVG_Turn", "AVG_GenTok"]
    for ds in DATASETS_DEFAULT:
        if ds in target_set:
            header +=[f"{ds}_acc(N)", f"{ds}_time(h)", f"{ds}_sec/samp", f"{ds}_peak", f"{ds}_dep(M)", f"{ds}_turn", f"{ds}_genTok"]
    print("\t".join(header))

    leaderboard =[]
    for model in models_sorted:
        avgs = model_avgs.get(model, AvgMetrics())
        s_avg_acc = "" if avgs.avg_acc is None else f"{avgs.avg_acc:.2f}"
        s_avg_peak = "" if avgs.avg_peak is None else f"{avgs.avg_peak:.0f}"
        
        s_avg_dep = "" if avgs.avg_dep is None else f"{avgs.avg_dep/1e6:.2f}"
        s_avg_sps = "" if avgs.avg_sec_per_sample is None else f"{avgs.avg_sec_per_sample:.3f}"
        
        s_avg_time_h = "" if avgs.avg_total_time is None else f"{avgs.avg_total_time/3600:.2f}"
        s_avg_turn = "" if avgs.avg_turn is None else f"{avgs.avg_turn:.2f}"
        s_avg_gen_tokens = "" if avgs.avg_total_generated_tokens is None else f"{avgs.avg_total_generated_tokens:.0f}"

        row =[model, s_avg_acc, s_avg_time_h, s_avg_sps, s_avg_peak, s_avg_dep, s_avg_turn, s_avg_gen_tokens]

        json_row: Dict[str, Any] = {
            "model_config": model,
            "avg_acc": avgs.avg_acc,
            "avg_peak": avgs.avg_peak,
            "avg_dep": avgs.avg_dep,
            "avg_dep_million": avgs.avg_dep / 1e6 if avgs.avg_dep is not None else None,
            "avg_sec_per_sample": avgs.avg_sec_per_sample,
            "avg_total_time_seconds": avgs.avg_total_time,
            "avg_total_time_hours": avgs.avg_total_time / 3600 if avgs.avg_total_time is not None else None,
            "avg_turn": avgs.avg_turn,
            "avg_total_generated_tokens": avgs.avg_total_generated_tokens,
        }

        for ds in DATASETS_DEFAULT:
            if ds not in target_set:
                continue
            m = results.get(model, {}).get(ds)
            
            json_row[f"{ds}_acc"] = None if not m else m.acc
            json_row[f"{ds}_peak"] = None if not m else m.peak
            json_row[f"{ds}_dep"] = None if not m else m.dep
            json_row[f"{ds}_dep_million"] = None if (not m or m.dep is None) else m.dep / 1e6
            json_row[f"{ds}_repeats"] = 0 if not m else m.num_repeats
            json_row[f"{ds}_selected_files"] =[] if (not m or not m.selected_files) else m.selected_files
            json_row[f"{ds}_sec_per_sample"] = None if not m else m.sec_per_sample
            json_row[f"{ds}_total_time_seconds"] = None if not m else m.total_time_seconds
            json_row[f"{ds}_total_time_hours"] = None if (not m or m.total_time_seconds is None) else m.total_time_seconds / 3600
            json_row[f"{ds}_avg_turn"] = None if not m else m.avg_turn
            json_row[f"{ds}_avg_total_generated_tokens"] = None if not m else m.avg_total_generated_tokens
            
            if not m:
                row +=["", "", "", "", "", "", ""]
            else:
                acc_str = "" if m.acc is None else f"{m.acc:.2f}"
                sps_str = "" if (m is None or m.sec_per_sample is None) else f"{m.sec_per_sample:.3f}"
                if m.num_repeats > 1:
                    acc_str += f"({m.num_repeats})"
                
                time_h_str = "" if m.total_time_seconds is None else f"{m.total_time_seconds/3600:.2f}"
                dep_m_str = "" if m.dep is None else f"{m.dep/1e6:.2f}"
                peak_str = "" if m.peak is None else f"{m.peak:.0f}"
                turn_str = "" if m.avg_turn is None else f"{m.avg_turn:.2f}"
                gen_tok_str = "" if m.avg_total_generated_tokens is None else f"{m.avg_total_generated_tokens:.0f}"

                row +=[acc_str, time_h_str, sps_str, peak_str, dep_m_str, turn_str, gen_tok_str]
        
        print("\t".join(row))
        leaderboard.append(json_row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] leaderboard json -> {out_path}")

    
    flat_wrongs =[]
    for model in models_sorted:
        if model not in all_wrong_cases:
            continue
        model_block = {"model_config": model, "datasets":[]}
        
        for ds in DATASETS_DEFAULT:
            if ds in all_wrong_cases[model]:
                ds_files = all_wrong_cases[model][ds]
                model_block["datasets"].append({
                    "dataset_name": ds,
                    "files": ds_files
                })
        
        if model_block["datasets"]:
            flat_wrongs.append(model_block)

    if flat_wrongs:
        os.makedirs(os.path.dirname(out_wrong_path) or ".", exist_ok=True)
        with open(out_wrong_path, "w", encoding="utf-8") as f:
            json.dump(flat_wrongs, f, ensure_ascii=False, indent=2)
        print(f"[Saved] wrong cases json -> {out_wrong_path}")
    else:
        print("[Info] No wrong cases found.")

if __name__ == "__main__":
    main()