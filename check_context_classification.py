import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

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
    POSTFIX_PROMPT,
)
from mpt_inference import (
    Decoder,
    mask_entities,
    format_prompt,
    decode_batch,
    get_pred,
    TOTAL_LLM_CALLS,
)


def set_tasks_BBQ():
    return [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_gender",
        "Race_x_SES",
        "Religion",
        "SES",
        "Sexual_orientation",
    ]


def load_evaluation_data(args, task_name):
    """Load and sample data for a specific task."""
    input_path = Path(args.data_dir) / args.dataset / f"{task_name}.jsonl"
    if not input_path.exists():
        print(f"Warning: File not found: {input_path}")
        return []

    results = [json.loads(line) for line in xopen(input_path, "r", encoding="utf-8")]

    if args.truncate:
        import random
        random.seed(args.random_seed)
        len_groups = 20
        example_groups = []
        for i in range(0, len(results), 4):
            if i + 3 < len(results):
                example_groups.append(results[i : i + 4])
        if len_groups < len(example_groups):
            selected_groups = random.sample(example_groups, len_groups)
            results = []
            for group in selected_groups:
                results.extend(group)
        else:
            results = []
            for group in example_groups:
                results.extend(group)
    elif args.sample_size and args.sample_size > 0:
        results = results[:args.sample_size]

    return results


def build_masked_prompt(args, item):
    """Build Turn 1 prompt for Context Sufficiency Check on entity-masked item."""
    masked_item = mask_entities(item)
    context = masked_item["context"]
    question = masked_item["question"]
    options = f"Options: (0): {masked_item['ans0']}; (1): {masked_item['ans1']}; (2): {masked_item['ans2']}"

    ask_prompt = CONSTRAINT_PROMPT_BBQ.format(
        question_prompt=QUESTION_PROMPT_BBQ,
        context=context,
        question=question,
        options=options
    )
    full_prompt_text = ask_prompt + LOGICAL_PROMPT_MASKED + POSTFIX_PROMPT
    formatted_prompt = format_prompt(
        args, full_prompt_text,
        sys_prompt=DEFAULT_SYS_PROMPT,
        id=args.summary_id
    )
    return full_prompt_text, formatted_prompt


def get_undetermined_index(item):
    """Extract the index (0, 1, or 2) corresponding to the undetermined option."""
    ainfo = item.get("answer_info", {})
    for k, v in ainfo.items():
        if len(v) > 1 and v[1] == "unknown":
            return int(k.replace("ans", ""))
    for idx in [0, 1, 2]:
        text = str(item.get(f"ans{idx}", "")).lower()
        if any(kw in text for kw in ["cannot be determined", "can't be determined", "undetermined", "unknown", "not enough information"]):
            return idx
    return 2


def is_classified_as_ambiguous(pred, item):
    """
    Check whether the prediction indicates insufficient context (Ambiguous -> Fast Path).
    Correctly handles permuted BBQ option positions (where undetermined can be 0, 1, or 2).
    """
    if pred is None or pred == -1:
        return True

    u_idx = get_undetermined_index(item)
    if pred == u_idx:
        return True

    ans_text = str(item.get(f"ans{pred}", "")).lower()
    undetermined_keywords = [
        "cannot be determined", "can't be determined",
        "undetermined", "not enough information",
        "unknown", "not determinable"
    ]
    return any(kw in ans_text for kw in undetermined_keywords)


def compute_metrics(total, true_amb, true_dis, fast_cnt, fast_amb, fast_dis, adapt_cnt, adapt_dis, adapt_amb):
    """Compute detailed distribution and percentage metrics for routing classification."""
    fast_pct_total = (fast_cnt / total * 100) if total > 0 else 0.0
    fast_amb_pct_fast = (fast_amb / fast_cnt * 100) if fast_cnt > 0 else 0.0
    fast_amb_recall = (fast_amb / true_amb * 100) if true_amb > 0 else 0.0
    fast_dis_pct_fast = (fast_dis / fast_cnt * 100) if fast_cnt > 0 else 0.0
    fast_dis_leak_rate = (fast_dis / true_dis * 100) if true_dis > 0 else 0.0

    adapt_pct_total = (adapt_cnt / total * 100) if total > 0 else 0.0
    adapt_dis_pct_adapt = (adapt_dis / adapt_cnt * 100) if adapt_cnt > 0 else 0.0
    adapt_dis_recall = (adapt_dis / true_dis * 100) if true_dis > 0 else 0.0
    adapt_amb_pct_adapt = (adapt_amb / adapt_cnt * 100) if adapt_cnt > 0 else 0.0
    adapt_amb_leak_rate = (adapt_amb / true_amb * 100) if true_amb > 0 else 0.0

    routing_acc = ((fast_amb + adapt_dis) / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "true_ambig": true_amb,
        "true_disambig": true_dis,
        "routing_acc": routing_acc,
        "fast_count": fast_cnt,
        "fast_pct_total": fast_pct_total,
        "fast_amb_count": fast_amb,
        "fast_amb_pct_fast": fast_amb_pct_fast,
        "fast_amb_recall": fast_amb_recall,
        "fast_dis_count": fast_dis,
        "fast_dis_pct_fast": fast_dis_pct_fast,
        "fast_dis_leak_rate": fast_dis_leak_rate,
        "adapt_count": adapt_cnt,
        "adapt_pct_total": adapt_pct_total,
        "adapt_dis_count": adapt_dis,
        "adapt_dis_pct_adapt": adapt_dis_pct_adapt,
        "adapt_dis_recall": adapt_dis_recall,
        "adapt_amb_count": adapt_amb,
        "adapt_amb_pct_adapt": adapt_amb_pct_adapt,
        "adapt_amb_leak_rate": adapt_amb_leak_rate,
    }


def evaluate_context_classification(args, models):
    """Run Context Sufficiency Check and evaluate routing distribution between Fast Path and Adaptive Path."""
    tasks = [args.task] if args.task else set_tasks_BBQ()

    overall_counts = {
        "total": 0,
        "true_ambig": 0,
        "true_disambig": 0,
        "fast_count": 0,
        "fast_ambig": 0,
        "fast_disambig_leak": 0,
        "adapt_count": 0,
        "adapt_disambig": 0,
        "adapt_ambig_leak": 0,
    }

    per_task_results = {}
    error_samples = []

    print("\n" + "=" * 115)
    print("EVALUATING CONTEXT SUFFICIENCY CLASSIFICATION & ROUTING DISTRIBUTION")
    print(f"Model: {args.models[0]} | Dataset: {args.dataset} | Tasks: {len(tasks)}")
    print("=" * 115)

    for task_name in tasks:
        data = load_evaluation_data(args, task_name)
        if not data:
            continue

        print(f"\nTask: {task_name:<22} | Samples: {len(data)}")

        # Build prompts for all samples in task (Masked Reasoning)
        prompts = []
        for it in data:
            _, f_prompt = build_masked_prompt(args, it)
            prompts.append(f_prompt)

        # Batch decode with model (or mock)
        if getattr(args, "mock", False):
            responses = []
            for it in data:
                cond = it.get("context_condition", "")
                if cond == "ambig":
                    u_idx = get_undetermined_index(it)
                    responses.append(f"The context lacks information to identify who performed the action. <answer> {u_idx} </answer>")
                else:
                    gold_label = it.get("label", 0)
                    responses.append(f"The context explicitly states the factual answer. <answer> {gold_label} </answer>")
        else:
            responses = decode_batch(models, args, prompts, phase="check_classification")

        task_total = len(data)
        task_true_amb = 0
        task_true_dis = 0
        task_fast_cnt = 0
        task_fast_amb = 0
        task_fast_dis_leak = 0
        task_adapt_cnt = 0
        task_adapt_dis = 0
        task_adapt_amb_leak = 0

        for it, resp in zip(data, responses):
            cond = it.get("context_condition", "unknown")
            is_ambig_gt = (cond == "ambig")
            if is_ambig_gt:
                task_true_amb += 1
            else:
                task_true_dis += 1

            pred = get_pred(resp)
            is_ambig_pred = is_classified_as_ambiguous(pred, it)

            if is_ambig_pred:
                task_fast_cnt += 1
                if is_ambig_gt:
                    task_fast_amb += 1
                else:
                    task_fast_dis_leak += 1
                    error_samples.append({
                        "task": task_name,
                        "type": "DISAMBIG_LEAKED_TO_FAST",
                        "condition": cond,
                        "context": it.get("context"),
                        "question": it.get("question"),
                        "gold": it.get("label"),
                        "pred": pred,
                        "response": resp,
                    })
            else:
                task_adapt_cnt += 1
                if not is_ambig_gt:
                    task_adapt_dis += 1
                else:
                    task_adapt_amb_leak += 1
                    error_samples.append({
                        "task": task_name,
                        "type": "AMBIG_LEAKED_TO_ADAPT",
                        "condition": cond,
                        "context": it.get("context"),
                        "question": it.get("question"),
                        "gold": it.get("label"),
                        "pred": pred,
                        "response": resp,
                    })

        # Accumulate overall totals
        overall_counts["total"] += task_total
        overall_counts["true_ambig"] += task_true_amb
        overall_counts["true_disambig"] += task_true_dis
        overall_counts["fast_count"] += task_fast_cnt
        overall_counts["fast_ambig"] += task_fast_amb
        overall_counts["fast_disambig_leak"] += task_fast_dis_leak
        overall_counts["adapt_count"] += task_adapt_cnt
        overall_counts["adapt_disambig"] += task_adapt_dis
        overall_counts["adapt_ambig_leak"] += task_adapt_amb_leak

        # Calculate task metrics
        task_metrics = compute_metrics(
            task_total, task_true_amb, task_true_dis,
            task_fast_cnt, task_fast_amb, task_fast_dis_leak,
            task_adapt_cnt, task_adapt_dis, task_adapt_amb_leak
        )
        per_task_results[task_name] = task_metrics

        print(
            f"  -> Routing Acc: {task_metrics['routing_acc']:.1f}% | "
            f"Fast: {task_metrics['fast_count']} ({task_metrics['fast_pct_total']:.1f}%) [Amb: {task_metrics['fast_amb_pct_fast']:.1f}%, Dis Leak: {task_metrics['fast_dis_pct_fast']:.1f}%] | "
            f"Adapt: {task_metrics['adapt_count']} ({task_metrics['adapt_pct_total']:.1f}%) [Dis: {task_metrics['adapt_dis_pct_adapt']:.1f}%, Amb Leak: {task_metrics['adapt_amb_pct_adapt']:.1f}%]"
        )

    # Calculate overall metrics
    overall_metrics = compute_metrics(
        overall_counts["total"],
        overall_counts["true_ambig"],
        overall_counts["true_disambig"],
        overall_counts["fast_count"],
        overall_counts["fast_ambig"],
        overall_counts["fast_disambig_leak"],
        overall_counts["adapt_count"],
        overall_counts["adapt_disambig"],
        overall_counts["adapt_ambig_leak"],
    )

    # ── Summary Report: Unified Routing Table ──
    header = (
        f"{'Task':<23} {'Samples':<8} {'Routing Acc':<13} "
        f"{'Fast Path: Total (% of all)'} | {'Amb %'}  | {'Disamb % (Leak)'}   "
        f"{'Adaptive Path: Total (% of all)'} | {'Disamb %'}  | {'Amb % (Leak)'}"
    )
    table_width = len(header)

    print("\n" + "=" * table_width)
    print("SUMMARY RESULTS: CONTEXT SUFFICIENCY ROUTING DISTRIBUTION")
    print("=" * table_width)
    print(header)
    print("-" * table_width)

    for task_name, m in per_task_results.items():
        f_total = f"{m['fast_count']} ({m['fast_pct_total']:.1f}%)"
        f_amb = f"{m['fast_amb_pct_fast']:.1f}%"
        f_dis = f"{m['fast_dis_pct_fast']:.1f}%"
        fast_cell = f"{f_total:^27} | {f_amb:^6} | {f_dis:^17}"

        a_total = f"{m['adapt_count']} ({m['adapt_pct_total']:.1f}%)"
        a_dis = f"{m['adapt_dis_pct_adapt']:.1f}%"
        a_amb = f"{m['adapt_amb_pct_adapt']:.1f}%"
        adapt_cell = f"{a_total:^31} | {a_dis:^9} | {a_amb:^14}"

        acc_str = f"{m['routing_acc']:.1f}%"
        print(f"{task_name:<23} {m['total']:<8} {acc_str:<13} {fast_cell}   {adapt_cell}")

    print("-" * table_width)

    tot_f_total = f"{overall_metrics['fast_count']} ({overall_metrics['fast_pct_total']:.1f}%)"
    tot_f_amb = f"{overall_metrics['fast_amb_pct_fast']:.1f}%"
    tot_f_dis = f"{overall_metrics['fast_dis_pct_fast']:.1f}%"
    tot_fast_cell = f"{tot_f_total:^27} | {tot_f_amb:^6} | {tot_f_dis:^17}"

    tot_a_total = f"{overall_metrics['adapt_count']} ({overall_metrics['adapt_pct_total']:.1f}%)"
    tot_a_dis = f"{overall_metrics['adapt_dis_pct_adapt']:.1f}%"
    tot_a_amb = f"{overall_metrics['adapt_amb_pct_adapt']:.1f}%"
    tot_adapt_cell = f"{tot_a_total:^31} | {tot_a_dis:^9} | {tot_a_amb:^14}"

    tot_acc_str = f"{overall_metrics['routing_acc']:.1f}%"
    print(f"{'OVERALL TOTAL':<23} {overall_metrics['total']:<8} {tot_acc_str:<13} {tot_fast_cell}   {tot_adapt_cell}")
    print("=" * table_width)

    # ── Key Metric Highlights ──
    print(f"\n>>> KEY METRIC HIGHLIGHTS:")
    print(f"    1. Total Samples Evaluated      : {overall_metrics['total']} (True Amb: {overall_metrics['true_ambig']}, True Disamb: {overall_metrics['true_disambig']})")
    print(f"    2. Fast Path Breakdown          : {overall_metrics['fast_count']} samples ({overall_metrics['fast_pct_total']:.2f}% of all)")
    print(f"       - Ambiguous Composition     : {overall_metrics['fast_amb_count']} ({overall_metrics['fast_amb_pct_fast']:.2f}% of Fast Path, Recall: {overall_metrics['fast_amb_recall']:.2f}%)")
    print(f"       - Disambiguated Leak (Error) : {overall_metrics['fast_dis_count']} ({overall_metrics['fast_dis_pct_fast']:.2f}% of Fast Path, Leak Rate: {overall_metrics['fast_dis_leak_rate']:.2f}%)")
    print(f"    3. Adaptive Path Breakdown      : {overall_metrics['adapt_count']} samples ({overall_metrics['adapt_pct_total']:.2f}% of all)")
    print(f"       - Disambiguated Composition : {overall_metrics['adapt_dis_count']} ({overall_metrics['adapt_dis_pct_adapt']:.2f}% of Adaptive Path, Recall: {overall_metrics['adapt_dis_recall']:.2f}%)")
    print(f"       - Ambiguous Leak (Error)     : {overall_metrics['adapt_amb_count']} ({overall_metrics['adapt_amb_pct_adapt']:.2f}% of Adaptive Path, Leak Rate: {overall_metrics['adapt_amb_leak_rate']:.2f}%)")
    correct_routing_count = overall_metrics['fast_amb_count'] + overall_metrics['adapt_dis_count']
    print(f"    4. Overall Routing Accuracy     : {overall_metrics['routing_acc']:.2f}% ({correct_routing_count}/{overall_metrics['total']} correctly routed)")
    print(f"    5. Total LLM Calls Executed     : {TOTAL_LLM_CALLS}\n")

    # Display leaked / error samples if requested
    if error_samples and args.show_errors:
        print(f"\nSample Leaked / Misclassified Samples ({len(error_samples)} total):")
        for s in error_samples[:args.show_errors]:
            print(f"\n[Task: {s['task']}] Leak Type: {s['type']} | Condition: {s['condition']} | Gold Label: {s['gold']}")
            print(f"Context: {s['context']}")
            print(f"Question: {s['question']}")
            print(f"Pred Option: {s['pred']} | Response Snippet:\n{s['response'][:200]}...")

    # Save to JSON if requested
    if args.output_json:
        os.makedirs(Path(args.output_json).parent, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "overall": {
                    "total_samples": overall_metrics["total"],
                    "true_ambiguous": overall_metrics["true_ambig"],
                    "true_disambiguated": overall_metrics["true_disambig"],
                    "routing_accuracy": overall_metrics["routing_acc"],
                    "fast_path": {
                        "total": overall_metrics["fast_count"],
                        "pct_of_all": overall_metrics["fast_pct_total"],
                        "ambiguous_count": overall_metrics["fast_amb_count"],
                        "ambiguous_pct_of_fast": overall_metrics["fast_amb_pct_fast"],
                        "ambiguous_recall": overall_metrics["fast_amb_recall"],
                        "disambiguated_leak_count": overall_metrics["fast_dis_count"],
                        "disambiguated_leak_pct_of_fast": overall_metrics["fast_dis_pct_fast"],
                        "disambiguated_leak_rate": overall_metrics["fast_dis_leak_rate"],
                    },
                    "adaptive_path": {
                        "total": overall_metrics["adapt_count"],
                        "pct_of_all": overall_metrics["adapt_pct_total"],
                        "disambiguated_count": overall_metrics["adapt_dis_count"],
                        "disambiguated_pct_of_adapt": overall_metrics["adapt_dis_pct_adapt"],
                        "disambiguated_recall": overall_metrics["adapt_dis_recall"],
                        "ambiguous_leak_count": overall_metrics["adapt_amb_count"],
                        "ambiguous_leak_pct_of_adapt": overall_metrics["adapt_amb_pct_adapt"],
                        "ambiguous_leak_rate": overall_metrics["adapt_amb_leak_rate"],
                    },
                    "total_llm_calls": TOTAL_LLM_CALLS,
                },
                "per_task": {
                    task_name: {
                        "total_samples": m["total"],
                        "true_ambiguous": m["true_ambig"],
                        "true_disambiguated": m["true_disambig"],
                        "routing_accuracy": m["routing_acc"],
                        "fast_path": {
                            "total": m["fast_count"],
                            "pct_of_all": m["fast_pct_total"],
                            "ambiguous_count": m["fast_amb_count"],
                            "ambiguous_pct_of_fast": m["fast_amb_pct_fast"],
                            "ambiguous_recall": m["fast_amb_recall"],
                            "disambiguated_leak_count": m["fast_dis_count"],
                            "disambiguated_leak_pct_of_fast": m["fast_dis_pct_fast"],
                            "disambiguated_leak_rate": m["fast_dis_leak_rate"],
                        },
                        "adaptive_path": {
                            "total": m["adapt_count"],
                            "pct_of_all": m["adapt_pct_total"],
                            "disambiguated_count": m["adapt_dis_count"],
                            "disambiguated_pct_of_adapt": m["adapt_dis_pct_adapt"],
                            "disambiguated_recall": m["adapt_dis_recall"],
                            "ambiguous_leak_count": m["adapt_amb_count"],
                            "ambiguous_leak_pct_of_adapt": m["adapt_amb_pct_adapt"],
                            "ambiguous_leak_rate": m["adapt_amb_leak_rate"],
                        },
                    }
                    for task_name, m in per_task_results.items()
                },
                "error_samples": error_samples,
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {args.output_json}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Check context sufficiency classification and routing distribution before full MPT inference"
    )

    parser.add_argument(
        "--models", type=str, nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"],
        help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--dataset", type=str, default="BBQ", choices=["BBQ"],
        help="Dataset name (BBQ)"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Specific task/category to evaluate (e.g. Age, Gender_identity). None = all 11 tasks."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing dataset folders"
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="Truncate BBQ to 20 groups of 4 (80 samples/task) matching standard MPT"
    )
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="Evaluate first N samples per task for quick testing"
    )
    parser.add_argument(
        "--random_seed", type=int, default=0,
        help="Random seed for sampling reproducibility"
    )
    parser.add_argument(
        "--inference_type", type=str, default="vllm", choices=["vllm", "openai"],
        help="Backend inference type"
    )
    parser.add_argument(
        "--chat_templates", action="store_true",
        help="Use tokenizer chat template"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=128,
        help="Max generated tokens for reasoning"
    )
    parser.add_argument(
        "--summary_id", type=int, default=0,
        help="Model index to use"
    )
    parser.add_argument(
        "--show_errors", type=int, default=3,
        help="Number of leaked/misclassified samples to display for debugging"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional path to save JSON metrics"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs to use for tensor parallelism in vLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85,
        help="GPU memory utilization ratio for vLLM"
    )
    parser.add_argument(
        "--max_model_len", type=int, default=32768,
        help="Maximum model context length for vLLM"
    )
    parser.add_argument(
        "--models_dir", type=str, default=None,
        help="Directory where models are downloaded/cached"
    )
    parser.add_argument(
        "--log_path", type=str, default=None,
        help="Directory to save logs"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run mock evaluation without loading actual LLM (useful for quick pipeline verification)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.mock:
        models = []
    else:
        models = [Decoder(args, m) for m in args.models]
    evaluate_context_classification(args, models)
