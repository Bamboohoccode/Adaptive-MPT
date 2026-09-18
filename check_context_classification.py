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


def evaluate_context_classification(args, models):
    """Run Context Sufficiency Check and compute classification accuracy against ground truth."""
    tasks = [args.task] if args.task else set_tasks_BBQ()
    
    total_metrics = {
        "total": 0,
        "true_ambig": 0,
        "true_disambig": 0,
        "pred_ambig_fast": 0,
        "pred_disambig_adapt": 0,
        "correct_fast_ambig": 0,
        "leaked_disambig_to_fast": 0,
        "correct_adapt_disambig": 0,
        "leaked_ambig_to_adapt": 0,
    }

    per_task_results = {}
    error_samples = []

    print("\n" + "=" * 90)
    print(f"EVALUATING CONTEXT SUFFICIENCY CLASSIFICATION (Masked Reasoning Turn 1)")
    print(f"Model: {args.models[0]} | Dataset: {args.dataset}")
    print("=" * 90)

    for task_name in tasks:
        data = load_evaluation_data(args, task_name)
        if not data:
            continue

        print(f"\nTask: {task_name:<22} | Samples: {len(data)}")

        # Build prompts for all samples in task
        prompts = []
        raw_prompts = []
        for it in data:
            p_text, f_prompt = build_masked_prompt(args, it)
            raw_prompts.append(p_text)
            prompts.append(f_prompt)

        # Batch decode with model
        if getattr(args, "mock", False):
            # Mock mode for testing pipeline logic without GPU
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
            responses = decode_batch(models, args, prompts)

        # Evaluate predictions
        task_metrics = {
            "total": len(data),
            "true_ambig": 0,
            "true_disambig": 0,
            "pred_ambig_fast": 0,
            "pred_disambig_adapt": 0,
            "correct_fast_ambig": 0,
            "leaked_disambig_to_fast": 0,
            "correct_adapt_disambig": 0,
            "leaked_ambig_to_adapt": 0,
        }

        for idx, (it, resp) in enumerate(zip(data, responses)):
            cond = it.get("context_condition", "unknown")
            is_ambig_gt = (cond == "ambig")
            pred = get_pred(resp)
            is_ambig_pred = is_classified_as_ambiguous(pred, it)

            if is_ambig_gt:
                task_metrics["true_ambig"] += 1
                total_metrics["true_ambig"] += 1
            else:
                task_metrics["true_disambig"] += 1
                total_metrics["true_disambig"] += 1

            if is_ambig_pred:
                task_metrics["pred_ambig_fast"] += 1
                total_metrics["pred_ambig_fast"] += 1
                if is_ambig_gt:
                    task_metrics["correct_fast_ambig"] += 1
                    total_metrics["correct_fast_ambig"] += 1
                else:
                    task_metrics["leaked_disambig_to_fast"] += 1
                    total_metrics["leaked_disambig_to_fast"] += 1
                    error_samples.append({
                        "task": task_name,
                        "type": "DISAMBIG_LEAKED_TO_FAST",
                        "context": it.get("context"),
                        "question": it.get("question"),
                        "pred": pred,
                        "response": resp,
                    })
            else:
                task_metrics["pred_disambig_adapt"] += 1
                total_metrics["pred_disambig_adapt"] += 1
                if not is_ambig_gt:
                    task_metrics["correct_adapt_disambig"] += 1
                    total_metrics["correct_adapt_disambig"] += 1
                else:
                    task_metrics["leaked_ambig_to_adapt"] += 1
                    total_metrics["leaked_ambig_to_adapt"] += 1

            total_metrics["total"] += 1

        # Calculate Task rates
        fast_n = task_metrics["pred_ambig_fast"]
        purity = (task_metrics["correct_fast_ambig"] / fast_n * 100) if fast_n > 0 else 0.0
        dis_total = task_metrics["true_disambig"]
        dis_leak_rate = (task_metrics["leaked_disambig_to_fast"] / dis_total * 100) if dis_total > 0 else 0.0
        acc = ((task_metrics["correct_fast_ambig"] + task_metrics["correct_adapt_disambig"]) / task_metrics["total"] * 100)

        per_task_results[task_name] = {
            **task_metrics,
            "purity": purity,
            "dis_leak_rate": dis_leak_rate,
            "accuracy": acc,
        }
        print(f"  -> Accuracy: {acc:.1f}% | Fast Purity: {purity:.1f}% | Disambig Leak into Fast: {task_metrics['leaked_disambig_to_fast']}/{dis_total} ({dis_leak_rate:.1f}%)")

    # ── Summary Report ──
    print("\n" + "=" * 90)
    print("SUMMARY RESULTS: CONTEXT SUFFICIENCY CLASSIFICATION")
    print("=" * 90)
    header = (
        f"{'Task':<23} {'Samples':<8} {'Accuracy':<10} "
        f"{'Fast Total':<12} {'Fast Purity':<14} {'Disamb Leak':<14}"
    )
    print(header)
    print("-" * 90)

    for task_name, m in per_task_results.items():
        fp_str = f"{m['pred_ambig_fast']} ({m['pred_ambig_fast']/m['total']*100:.1f}%)"
        leak_str = f"{m['leaked_disambig_to_fast']} ({m['dis_leak_rate']:.1f}%)"
        print(
            f"{task_name:<23} {m['total']:<8} {m['accuracy']:.1f}%     "
            f"{fp_str:<12} {m['purity']:.1f}%        {leak_str:<14}"
        )

    print("-" * 90)
    tot = total_metrics["total"]
    overall_acc = ((total_metrics["correct_fast_ambig"] + total_metrics["correct_adapt_disambig"]) / tot * 100) if tot > 0 else 0.0
    tot_fast = total_metrics["pred_ambig_fast"]
    overall_purity = (total_metrics["correct_fast_ambig"] / tot_fast * 100) if tot_fast > 0 else 0.0
    tot_dis = total_metrics["true_disambig"]
    overall_leak = (total_metrics["leaked_disambig_to_fast"] / tot_dis * 100) if tot_dis > 0 else 0.0

    print(
        f"{'OVERALL TOTAL':<23} {tot:<8} {overall_acc:.1f}%     "
        f"{tot_fast:<12} {overall_purity:.1f}%        {total_metrics['leaked_disambig_to_fast']} ({overall_leak:.1f}%)"
    )
    print("=" * 90)

    print(f"\n>>> KEY METRIC HIGHLIGHTS:")
    print(f"    1. Overall Routing Accuracy : {overall_acc:.2f}%")
    print(f"    2. Fast Path Purity         : {overall_purity:.2f}% (Target: near 100% Ambig in Fast Path)")
    print(f"    3. Disambig Leak Rate       : {overall_leak:.2f}% ({total_metrics['leaked_disambig_to_fast']}/{tot_dis} disambig leaked to fast path)")
    print(f"    4. Ambig Recall to Fast     : {total_metrics['correct_fast_ambig']/total_metrics['true_ambig']*100:.2f}%\n")

    # Print error sample if requested
    if error_samples and args.show_errors:
        print(f"\nSample Disambig Leaked to Fast Path ({len(error_samples)} total):")
        for s in error_samples[:args.show_errors]:
            print(f"\n[Task: {s['task']}] Context: {s['context']}")
            print(f"Question: {s['question']}")
            print(f"Pred: {s['pred']} | Response Snippet:\n{s['response'][:200]}...")

    # Save to JSON if requested
    if args.output_json:
        os.makedirs(Path(args.output_json).parent, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "overall": {
                    "accuracy": overall_acc,
                    "fast_purity": overall_purity,
                    "disambig_leak_rate": overall_leak,
                    **total_metrics
                },
                "per_task": per_task_results,
                "error_samples": error_samples,
            }, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to {args.output_json}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Check context sufficiency classification accuracy before full MPT inference")

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
        help="Max generated tokens for Turn 1 reasoning"
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
