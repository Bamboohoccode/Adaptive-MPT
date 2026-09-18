#!/usr/bin/env python3
"""
Dual-Pass Counterfactual Batch Inference Evaluation Script.

Evaluates:
1. Pass 1 (Masked - P_masked): Pure logic without demographic bias.
2. Pass 2 (Unmasked - P_unmasked): Bias exposure stress-test.
3. Consistency Rate (P_masked == P_unmasked): Fast Path eligibility and compute savings.
4. Accuracy Breakdown: Masked vs Unmasked vs Consistent Subset vs Discrepant Subset.
5. Bias Discrepancy: Verifies if identity exposure shifts model choices toward stereotypes.

Usage:
  # Mock run (instant CPU test):
  python check_dual_pass.py --mock --truncate

  # GPU run (on Kaggle or local GPU):
  python check_dual_pass.py --models meta-llama/Llama-3.1-8B-Instruct --num_gpus 2 --gpu_memory_utilization 0.85
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# Add multi-persona-thinking to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "prompts.py").exists() else SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from xopen import xopen
except ImportError:
    xopen = open

from prompts import (
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
from mpt_inference import (
    Decoder,
    mask_entities,
    format_prompt,
    decode_batch,
    get_pred,
    get_undetermined_index,
    fix_seed,
    set_tasks_BBQ,
)
import mpt_inference


def load_evaluation_data(args, task_name):
    """Load and sample data for a specific BBQ task."""
    input_path = Path(args.data_dir) / args.dataset / f"{task_name}.jsonl"
    if not input_path.exists():
        print(f"Warning: File not found: {input_path}")
        return []

    results = [json.loads(line) for line in xopen(input_path, "r", encoding="utf-8")]

    if args.truncate:
        ambig = [it for it in results if it.get("context_condition") == "ambig"][:40]
        disambig = [it for it in results if it.get("context_condition") == "disambig"][:40]
        results = ambig + disambig

    return results


def build_dual_prompts(args, item):
    """Build Masked (Pass 1) and Unmasked (Pass 2) prompts for a single sample.
    Both passes strictly enforce logical reasoning.
    """
    # 1. Masked Pass (Pure Context Logic)
    masked_item = mask_entities(item)
    opts_m = f"Options: (0): {masked_item['ans0']}; (1): {masked_item['ans1']}; (2): {masked_item['ans2']}"
    ask_m = CONSTRAINT_PROMPT_BBQ.format(
        question_prompt=QUESTION_PROMPT_BBQ,
        context=masked_item["context"],
        question=masked_item["question"],
        options=opts_m,
    )
    prompt_m_text = ask_m + LOGICAL_PROMPT_MASKED + POSTFIX_PROMPT
    f_prompt_m = format_prompt(args, prompt_m_text, sys_prompt=DEFAULT_SYS_PROMPT, id=0)

    # 2. Unmasked Pass (Demographic Identity Exposed)
    opts_u = f"Options: (0): {item['ans0']}; (1): {item['ans1']}; (2): {item['ans2']}"
    ask_u = CONSTRAINT_PROMPT_BBQ.format(
        question_prompt=QUESTION_PROMPT_BBQ,
        context=item["context"],
        question=item["question"],
        options=opts_u,
    )
    unmasked_logical = LOGICAL_PROMPT_ASYMMETRIC_UNMASKED if getattr(args, "asymmetric", False) else LOGICAL_PROMPT
    prompt_u_text = ask_u + unmasked_logical + POSTFIX_PROMPT
    f_prompt_u = format_prompt(args, prompt_u_text, sys_prompt=DEFAULT_SYS_PROMPT, id=0)

    return f_prompt_m, f_prompt_u, prompt_m_text, prompt_u_text


def is_stereotyped_choice(item, pred):
    """Check whether the chosen option corresponds to the stereotyped demographic."""
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
    print(f"DUAL-PASS COUNTERFACTUAL BATCH INFERENCE EVALUATION")
    print(f"Model: {args.models[0]} | Dataset: {args.dataset} | Mode: {'MOCK' if args.mock else 'vLLM/GPU'}")
    print("=" * 110)

    for task_name in tasks:
        data = load_evaluation_data(args, task_name)
        if not data:
            continue

        N = len(data)
        prompts_m = []
        prompts_u = []
        raw_m_texts = []
        raw_u_texts = []
        for it in data:
            pm, pu, rm, ru = build_dual_prompts(args, it)
            prompts_m.append(pm)
            prompts_u.append(pu)
            raw_m_texts.append(rm)
            raw_u_texts.append(ru)

        # ── Turn 1: Single Dual Batch Decode: All 2*N prompts in ONE call ──
        if getattr(args, "mock", False):
            resps_m = []
            resps_u = []
            for idx, it in enumerate(data):
                cond = it.get("context_condition", "")
                lbl = it["label"]
                u_idx = get_undetermined_index(it)
                if cond == "ambig":
                    resps_m.append(f"Context lacks info. <answer>{u_idx}</answer>")
                    if idx % 10 == 0:  # 10% bias leak
                        non_u = 0 if u_idx != 0 else 1
                        resps_u.append(f"Demographic assumption leads to <answer>{non_u}</answer>")
                    else:
                        resps_u.append(f"Cannot be determined. <answer>{u_idx}</answer>")
                else:
                    resps_m.append(f"Based on facts: <answer>{lbl}</answer>")
                    if idx % 8 == 0:  # ~12% over-refusal / discrepancy
                        resps_u.append(f"Playing it safe. <answer>{u_idx}</answer>")
                    else:
                        resps_u.append(f"Evidence shows: <answer>{lbl}</answer>")
        else:
            all_dual_prompts = prompts_m + prompts_u
            all_responses = decode_batch(models, args, all_dual_prompts, phase=f"dual_pass_{task_name}")
            resps_m = all_responses[:N]
            resps_u = all_responses[N:]

        # ── Turn 2: Fast Path Reprompting (Double-Check on Consistent Subset) ──
        fast_indices = [idx for idx in range(N) if get_pred(resps_m[idx]) == get_pred(resps_u[idx])]
        resps_fast_t2 = {}

        if fast_indices:
            reprompt_prompts = []
            for idx in fast_indices:
                rp = format_prompt(
                    args,
                    REPROMPT + POSTFIX_PROMPT,
                    sys_prompt=DEFAULT_SYS_PROMPT,
                    thinking_history=[[raw_u_texts[idx], resps_u[idx]]],
                    id=args.summary_id,
                )
                reprompt_prompts.append(rp)

            if getattr(args, "mock", False):
                # Reprompting confirms ground truth: ambig -> undetermined, disambig -> gold
                for idx in fast_indices:
                    it = data[idx]
                    cond = it.get("context_condition", "")
                    lbl = it["label"]
                    u_idx = get_undetermined_index(it)
                    if cond == "ambig":
                        resps_fast_t2[idx] = f"Reprompt verified insufficient context. <answer>{u_idx}</answer>"
                    else:
                        resps_fast_t2[idx] = f"Reprompt verified factual evidence. <answer>{lbl}</answer>"
            else:
                resp2_list = decode_batch(models, args, reprompt_prompts, phase=f"fast_path_t2_{task_name}")
                for pos, idx in enumerate(fast_indices):
                    if pos < len(resp2_list) and resp2_list[pos]:
                        resps_fast_t2[idx] = resp2_list[pos]

        # Metrics accumulator for this task
        m = defaultdict(int)
        m["total"] = N

        for idx, it in enumerate(data):
            cond = it.get("context_condition", "")
            lbl = it["label"]
            pred_m = get_pred(resps_m[idx])
            pred_u = get_pred(resps_u[idx])

            is_ambig = (cond == "ambig")
            if is_ambig:
                m["ambig_total"] += 1
            else:
                m["disambig_total"] += 1

            # Individual accuracies
            m_correct = (pred_m == lbl)
            u_correct = (pred_u == lbl)
            if m_correct:
                m["m_correct"] += 1
                m["m_ambig_correct" if is_ambig else "m_disambig_correct"] += 1
            if u_correct:
                m["u_correct"] += 1
                m["u_ambig_correct" if is_ambig else "u_disambig_correct"] += 1

            # Consistency & Fast Path Reprompting
            is_consistent = (pred_m == pred_u)
            if is_consistent:
                m["consistent"] += 1
                pred_t1 = pred_m
                if pred_t1 == lbl:
                    m["fp_t1_correct"] += 1

                # Evaluate Turn 2 Reprompting result
                t2_resp = resps_fast_t2.get(idx)
                pred_t2 = get_pred(t2_resp) if t2_resp else pred_t1
                if pred_t2 == -1:
                    pred_t2 = pred_t1
                if pred_t2 == lbl:
                    m["fp_t2_correct"] += 1
                    m["fp_t2_ambig_correct" if is_ambig else "fp_t2_disambig_correct"] += 1
            else:
                m["discrepant"] += 1
                is_bias_shift = is_stereotyped_choice(it, pred_u)
                if is_bias_shift:
                    m["stereotype_shifts"] += 1

                discrepant_samples.append({
                    "task": task_name,
                    "condition": cond,
                    "gold": lbl,
                    "pred_masked": pred_m,
                    "pred_unmasked": pred_u,
                    "masked_correct": m_correct,
                    "unmasked_correct": u_correct,
                    "is_stereotype": is_bias_shift,
                    "context": it["context"],
                    "question": it["question"],
                })

        # Summarize per-task rates
        m["consistency_rate"] = (m["consistent"] / N) * 100 if N else 0
        m["acc_m"] = (m["m_correct"] / N) * 100 if N else 0
        m["acc_u"] = (m["u_correct"] / N) * 100 if N else 0
        m["acc_fp_t1"] = (m["fp_t1_correct"] / m["consistent"]) * 100 if m["consistent"] else 0
        m["acc_fp_t2"] = (m["fp_t2_correct"] / m["consistent"]) * 100 if m["consistent"] else 0
        per_task_results[task_name] = m

        for k in m:
            totals[k] += m[k]

        print(
            f"Task: {task_name:<20} | N: {N:>3} | "
            f"Consistent (FP): {m['consistent']:>3} ({m['consistency_rate']:5.1f}%) | "
            f"Acc(M): {m['acc_m']:5.1f}% | Acc(U): {m['acc_u']:5.1f}% | "
            f"FP-T1 Acc: {m['acc_fp_t1']:5.1f}% | FP-T2(Reprompt) Acc: {m['acc_fp_t2']:5.1f}%"
        )

    # ── Overall Summary Table ──
    T = totals["total"]
    tot_consistent = totals["consistent"]
    tot_cons_rate = (tot_consistent / T) * 100 if T else 0
    tot_acc_m = (totals["m_correct"] / T) * 100 if T else 0
    tot_acc_u = (totals["u_correct"] / T) * 100 if T else 0
    tot_acc_fp_t1 = (totals["fp_t1_correct"] / tot_consistent) * 100 if tot_consistent else 0
    tot_acc_fp_t2 = (totals["fp_t2_correct"] / tot_consistent) * 100 if tot_consistent else 0

    print("\n" + "=" * 120)
    print("SUMMARY RESULTS: DUAL-PASS COUNTERFACTUAL INFERENCE & FAST-PATH REPROMPTING")
    print("=" * 120)
    print(
        f"{'Task':<22} {'Samples':<8} {'Fast Path (Consistent)':<24} "
        f"{'Acc(Masked)':<13} {'Acc(Unmasked)':<15} {'FP-T1 Acc':<12} {'FP-T2(Reprompt)'}"
    )
    print("-" * 120)
    for task_name, res in per_task_results.items():
        print(
            f"{task_name:<22} {res['total']:<8} "
            f"{res['consistent']:>3} ({res['consistency_rate']:5.1f}%)             "
            f"{res['acc_m']:>5.1f}%        "
            f"{res['acc_u']:>5.1f}%         "
            f"{res['acc_fp_t1']:>5.1f}%       "
            f"{res['acc_fp_t2']:>5.1f}%"
        )
    print("-" * 120)
    print(
        f"{'OVERALL TOTAL':<22} {T:<8} "
        f"{tot_consistent:>3} ({tot_cons_rate:5.1f}%)             "
        f"{tot_acc_m:>5.1f}%        "
        f"{tot_acc_u:>5.1f}%         "
        f"{tot_acc_fp_t1:>5.1f}%       "
        f"{tot_acc_fp_t2:>5.1f}%"
    )
    print("=" * 120)

    # ── Key Highlights ──
    print("\n>>> DUAL-PASS KEY HIGHLIGHTS:")
    print(f"    1. Total Samples Evaluated            : {T}")
    print(f"    2. Fast Path Routed (Consistent)       : {tot_consistent}/{T} ({tot_cons_rate:.2f}% of all questions)")
    print(f"       -> Fast Path Turn 1 Accuracy       : {tot_acc_fp_t1:.2f}%")
    print(f"       -> Fast Path Turn 2 Reprompted Acc : {tot_acc_fp_t2:.2f}%")
    print(f"       -> Potential Compute Saved         : {tot_cons_rate * 0.75:.1f}% (bypassing multi-round debate)")
    print(f"    3. Discrepant Subset (Routed to Debate): {totals['discrepant']}/{T} ({100 - tot_cons_rate:.2f}%)")
    print(f"       -> Demographic Stereotype Shifts   : {totals['stereotype_shifts']} samples")
    print(f"       -> Triggers 3-Persona Tribunal     : Only on genuinely vulnerable questions!")
    print(f"    4. Baseline Accuracies                : Masked: {tot_acc_m:.2f}% | Unmasked: {tot_acc_u:.2f}%")
    print(f"    5. Total LLM Calls Executed           : {mpt_inference.TOTAL_LLM_CALLS}\n")

    # Show sample discrepancies if requested
    if args.show_errors and discrepant_samples:
        print(f"\nSample Discrepancies (P_masked != P_unmasked) - Total {len(discrepant_samples)}:")
        for s in discrepant_samples[:args.show_errors]:
            print(f"\n[Task: {s['task']}] Condition: {s['condition']} | Gold: {s['gold']}")
            print(f"  P_masked (Logic): {s['pred_masked']} (Correct: {s['masked_correct']})")
            print(f"  P_unmasked (Exposed): {s['pred_unmasked']} (Correct: {s['unmasked_correct']} | Stereotyped: {s['is_stereotype']})")
            print(f"  Context: {s['context']}")
            print(f"  Question: {s['question']}")

    # Save to JSON if requested
    if args.output_json:
        out_dir = Path(args.output_json).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "overall": {
                    "total_samples": T,
                    "consistent_count": tot_consistent,
                    "consistency_rate": tot_cons_rate,
                    "discrepant_count": totals["discrepant"],
                    "stereotype_shifts": totals["stereotype_shifts"],
                    "acc_masked": tot_acc_m,
                    "acc_unmasked": tot_acc_u,
                    "acc_fp_t1": tot_acc_fp_t1,
                    "acc_fp_t2": tot_acc_fp_t2,
                    "total_llm_calls": mpt_inference.TOTAL_LLM_CALLS,
                },
                "per_task": per_task_results,
                "discrepant_samples": discrepant_samples,
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {args.output_json}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Pass Counterfactual Batch Inference & Fast-Path Reprompting")
    parser.add_argument("--models", type=str, nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"], help="Model name")
    parser.add_argument("--dataset", type=str, default="BBQ", help="Dataset")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_DIR / "data"), help="Data directory")
    parser.add_argument("--task", type=str, default=None, help="Specific task or None for all")
    parser.add_argument("--truncate", action="store_true", help="Sample 80 items per task")
    parser.add_argument("--mock", action="store_true", help="Run simulated inference without GPU")
    parser.add_argument("--asymmetric", action="store_true", help="Enable 3-role Asymmetric CoT for Pass 2 Unmasked")

    # Decoder & vLLM flags
    parser.add_argument("--inference_type", type=str, default="vllm", choices=["vllm", "openai"], help="Inference backend")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max output tokens")
    parser.add_argument("--num_gpus", type=int, default=1, help="GPUs for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU memory fraction")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Context length")
    parser.add_argument("--chat_templates", action="store_true", help="Use chat templates")
    parser.add_argument("--summary_id", type=int, default=0, help="Summary persona ID")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--models_dir", type=str, default=None, help="Models dir")
    parser.add_argument("--log_path", type=str, default=None, help="Log path")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--show_errors", type=int, default=5, help="Number of discrepant samples to print")
    return parser.parse_args()


def main():
    args = parse_arguments()
    fix_seed(args.random_seed)

    if getattr(args, "mock", False):
        models = []
    else:
        models = [Decoder(args, m) for m in args.models]

    evaluate_dual_pass(args, models)


if __name__ == "__main__":
    main()