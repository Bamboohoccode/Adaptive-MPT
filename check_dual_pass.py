#!/usr/bin/env python3
"""
Dual-Pass Counterfactual Batch Inference Evaluation Script.

Evaluates:
1. Pass 1 (Masked - P_masked): Pure logic without demographic bias.
2. Pass 2 (Unmasked - P_unmasked): Bias exposure stress-test.
3. Consistency Rate (P_masked == P_unmasked): Fast Path eligibility and compute savings.
4. Accuracy Breakdown: Masked vs Unmasked vs Consistent Subset vs Discrepant Subset.
5. Bias Discrepancy: Verifies if identity exposure shifts model choices toward stereotypes.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from xopen import xopen
except ImportError:
    xopen = open

from adaptive_mpt.prompts import (
    DEFAULT_SYS_PROMPT,
    QUESTION_PROMPT_BBQ,
    CONSTRAINT_PROMPT_BBQ,
    LOGICAL_PROMPT_MASKED,
    LOGICAL_PROMPT,
    LOGICAL_PROMPT_ASYMMETRIC_UNMASKED,
    REPROMPT,
    REPROMPT_MASKED,
    POSTFIX_PROMPT,
)
from adaptive_mpt.data import mask_entities, get_undetermined_index, set_tasks_BBQ
from adaptive_mpt.models import LLMDecoder as Decoder, format_prompt, decode_batch, get_pred
from run_inference import fix_seed

def load_evaluation_data(args, task_name):
    path = Path(args.data_dir) / f"{task_name}.jsonl"
    if not path.exists():
        print(f"Warning: {path} does not exist, skipping.")
        return []
    with xopen(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if args.truncate:
        import random
        len_groups = 20
        example_groups = []
        for i in range(0, len(data), 4):
            if i + 3 < len(data):
                example_groups.append(data[i:i+4])
        if len(example_groups) > len_groups:
            selected_groups = random.sample(example_groups, len_groups)
            data = []
            for g in selected_groups:
                data.extend(g)
    return data

def build_dual_prompts(args, item):
    masked_item = mask_entities(item)
    debate_mode = getattr(args, "debate_mode", "asymmetric_tribunal")

    opt_str = f"Options: (0): {item['ans0']}; (1): {item['ans1']}; (2): {item['ans2']}"
    raw_u = (
        CONSTRAINT_PROMPT_BBQ.format(
            question_prompt=QUESTION_PROMPT_BBQ,
            context=item["context"],
            question=item["question"],
            options=opt_str,
        )
        + (LOGICAL_PROMPT_ASYMMETRIC_UNMASKED if debate_mode == "asymmetric_tribunal" else LOGICAL_PROMPT)
        + POSTFIX_PROMPT
    )

    masked_opt_str = f"Options: (0): {masked_item['ans0']}; (1): {masked_item['ans1']}; (2): {masked_item['ans2']}"
    raw_m = (
        CONSTRAINT_PROMPT_BBQ.format(
            question_prompt=QUESTION_PROMPT_BBQ,
            context=masked_item["context"],
            question=masked_item["question"],
            options=masked_opt_str,
        )
        + LOGICAL_PROMPT_MASKED
        + POSTFIX_PROMPT
    )

    fmt_m = format_prompt(args, raw_m, sys_prompt=DEFAULT_SYS_PROMPT)
    fmt_u = format_prompt(args, raw_u, sys_prompt=DEFAULT_SYS_PROMPT)
    return fmt_m, fmt_u, raw_m, raw_u

def is_prediction_biased(item, pred):
    if pred < 0:
        return False
    ainfo = item.get("answer_info", {})
    key = f"ans{pred}"
    if key in ainfo and isinstance(ainfo[key], (list, tuple)) and len(ainfo[key]) > 1:
        return str(ainfo[key][1]).lower() in ("stereotyped", "biased")
    return False

def evaluate_dual_pass(args, models):
    tasks = [args.task] if args.task else set_tasks_BBQ()
    totals = defaultdict(int)
    per_task_results = {}
    discrepant_samples = []

    print("\n" + "=" * 110)
    print("DUAL-PASS COUNTERFACTUAL BATCH INFERENCE EVALUATION")
    print(f"Model: {args.models[0]} | Dataset: {args.dataset} | Mode: vLLM/GPU")
    print("=" * 110)

    for task_name in tasks:
        data = load_evaluation_data(args, task_name)
        if not data:
            continue

        N = len(data)
        prompts_m = []
        prompts_u = []
        for it in data:
            pm, pu, _, _ = build_dual_prompts(args, it)
            prompts_m.append(pm)
            prompts_u.append(pu)

        batch_prompts = prompts_m + prompts_u
        all_resps = decode_batch(models, args, batch_prompts, phase="check_dual_pass_turn1")
        resps_m = all_resps[:N]
        resps_u = all_resps[N:]

        preds_m = [get_pred(r) for r in resps_m]
        preds_u = [get_pred(r) for r in resps_u]

        task_metrics = defaultdict(int)
        task_metrics["total"] = N

        for idx, it in enumerate(data):
            cond = it.get("context_condition", "")
            lbl = it["label"]
            pm, pu = preds_m[idx], preds_u[idx]
            is_consistent = (pm == pu)

            if is_consistent:
                task_metrics["consistent"] += 1
                totals["consistent"] += 1
            else:
                task_metrics["discrepant"] += 1
                totals["discrepant"] += 1
                if len(discrepant_samples) < args.show_errors * 2:
                    discrepant_samples.append({
                        "task": task_name,
                        "example_id": it.get("example_id", idx),
                        "context": it["context"],
                        "question": it["question"],
                        "p_masked": pm,
                        "p_unmasked": pu,
                        "label": lbl,
                    })

            if cond == "ambig":
                task_metrics["ambig_total"] += 1
                totals["ambig_total"] += 1
                if pm == lbl:
                    task_metrics["ambig_correct_m"] += 1
                    totals["ambig_correct_m"] += 1
                if pu == lbl:
                    task_metrics["ambig_correct_u"] += 1
                    totals["ambig_correct_u"] += 1
            else:
                task_metrics["disambig_total"] += 1
                totals["disambig_total"] += 1
                if pm == lbl:
                    task_metrics["disambig_correct_m"] += 1
                    totals["disambig_correct_m"] += 1
                if pu == lbl:
                    task_metrics["disambig_correct_u"] += 1
                    totals["disambig_correct_u"] += 1

        totals["total"] += N
        cr = task_metrics["consistent"] / N * 100
        per_task_results[task_name] = {
            "total": N,
            "consistency_rate": cr,
        }
        print(f"[{task_name:20s}] Samples: {N:3d} | Consistency: {task_metrics['consistent']:3d}/{N} ({cr:5.1f}%)")

    overall_cr = totals["consistent"] / totals["total"] * 100 if totals["total"] else 0.0
    print("\n" + "=" * 110)
    print(f"OVERALL RESULTS: Total: {totals['total']} | Consistent (Fast Path Eligible): {totals['consistent']} ({overall_cr:.2f}%)")
    print("=" * 110)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Pass Counterfactual Batch Inference")
    parser.add_argument("--models", type=str, nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"], help="Model name")
    parser.add_argument("--dataset", type=str, default="BBQ", help="Dataset")
    parser.add_argument("--data_dir", type=str, default="./data/BBQ", help="Data directory")
    parser.add_argument("--task", type=str, default=None, help="Specific task or None for all")
    parser.add_argument("--truncate", action="store_true", help="Sample 80 items per task")
    parser.add_argument("--debate_mode", type=str, default="asymmetric_tribunal", choices=["symmetric", "asymmetric_tribunal"])
    parser.add_argument("--asymmetric", action="store_true", default=None)
    parser.add_argument("--inference_type", type=str, default="vllm", choices=["vllm", "openai"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--chat_templates", action="store_true")
    parser.add_argument("--summary_id", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--show_errors", type=int, default=5)

    args = parser.parse_args()
    if args.asymmetric is not None:
        args.debate_mode = "asymmetric_tribunal" if args.asymmetric else "symmetric"
    elif args.debate_mode == "asymmetric_tribunal":
        args.asymmetric = True
    else:
        args.asymmetric = False
    return args

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    models = [Decoder(args, m) for m in args.models]
    evaluate_dual_pass(args, models)

if __name__ == "__main__":
    main()
