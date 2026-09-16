import argparse
import json
import os
from cross_encoder import CrossEncoderRouter,load_ce_cache,save_ce_cache
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

try:
    from xopen import xopen
except ImportError:
    xopen = open

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEFAULT_DEVICE = "cpu"

# ── Ground truth category to Cross-Encoder bias labels mapping ──
BBQ_CATEGORY_MAP = {
    "Age": ["age"],
    "Disability_status": ["disability"],
    "Gender_identity": ["gender"],
    "Nationality": ["nationality"],
    "Physical_appearance": ["appearance"],
    "Race_ethnicity": ["race"],
    "Race_x_gender": ["race", "gender"],
    "Race_x_SES": ["race", "socioeconomic"],
    "Religion": ["religion"],
    "SES": ["socioeconomic"],
    "Sexual_orientation": ["sexuality"],
}

STEREOSET_CATEGORY_MAP = {
    "gender": ["gender"],
    "race": ["race"],
    "religion": ["religion"],
    "profession": ["profession"],
}


import random

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


def set_tasks_stereoset():
    return [
        "gender_inter",
        "gender_intra",
        "race_inter",
        "race_intra",
        "religion_inter",
        "religion_intra",
        "profession_inter",
        "profession_intra",
    ]


def get_answer(data, args):
    """Extract context condition / ground truth neutral indicator."""
    if args.dataset == "BBQ":
        return data.get("context_condition", "")
    elif args.dataset == "stereoset":
        # In StereoSet, label 0 is "Not enough information" (unrelated/neutral)
        return data.get("label") == 0
    return False


def load_dataset(args):
    """Load dataset files for BBQ or StereoSet matching MPT loading and truncate logic."""
    if args.dataset == "BBQ":
        tasks = set_tasks_BBQ()
    elif args.dataset == "stereoset":
        tasks = set_tasks_stereoset()
    else:
        tasks = []

    all_data = []
    print(f"Loading {args.dataset} dataset from {args.data_dir}/{args.dataset} (truncate={args.truncate})...")
    for task in tasks:
        input_path = Path(args.data_dir) / args.dataset / f"{task}.jsonl"
        if not input_path.exists():
            print(f"Warning: File not found: {input_path}")
            continue

        results = [json.loads(line) for line in xopen(input_path, "r", encoding="utf-8")]

        if args.truncate:
            if args.dataset == "BBQ":
                len_groups = 20
                original_count = len(results)
                example_groups = []
                for i in range(0, len(results), 4):
                    if i + 3 < len(results):
                        example_groups.append(results[i : i + 4])
                if len(example_groups) > len_groups:
                    selected_groups = random.sample(example_groups, len_groups)
                    results = []
                    for group in selected_groups:
                        results.extend(group)
                    print(f"Task: {task}, Selected groups: {len_groups}, Sampled samples: {len(results)} (orig: {original_count})")
                else:
                    results = []
                    for group in example_groups:
                        results.extend(group)
                    print(f"Task: {task}, Using all groups: {len(example_groups)}, Total samples: {len(results)}")

            elif args.dataset == "stereoset":
                len_samples = 100
                original_count = len(results)
                if len(results) > len_samples:
                    results = random.sample(results, len_samples)
                print(f"Task: {task}, Selected samples: {len(results)} (orig: {original_count})")

        all_data.extend(results)

    print(f"Total samples loaded across all {len(tasks)} tasks: {len(all_data)}")
    return all_data


def compute_metrics_for_kc(data, all_scores, args, top_c: int, top_k: int):
    """Compute routing metrics for a specific (top_c, top_k) pair."""
    total = len(data)
    fast_indices = []
    adaptive_indices = []
    persona_hits = 0

    category_map = BBQ_CATEGORY_MAP if args.dataset == "BBQ" else STEREOSET_CATEGORY_MAP

    for i, (line, scores) in enumerate(zip(data, all_scores)):
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_c_labels = [label for label, _ in ranked[:top_c]]
        is_fast = "neutral" in top_c_labels
        bias_labels = [l for l, _ in ranked if l not in ("neutral", "uncertainty")][:top_k]

        if is_fast:
            fast_indices.append(i)
        else:
            adaptive_indices.append(i)
            # Evaluate persona recall in Adaptive Path:
            # Does top_k contain the true ground-truth category of this question?
            cat = line.get("category", "")
            target_labels = category_map.get(cat, [])
            if any(tl in bias_labels for tl in target_labels):
                persona_hits += 1

    num_fast_path = len(fast_indices)
    num_adaptive_path = len(adaptive_indices)

    if args.dataset == "BBQ":
        num_neutral_in_fastpath = sum(
            1 for idx in fast_indices if str(get_answer(data[idx], args)).startswith("amb")
        )
    elif args.dataset == "stereoset":
        num_neutral_in_fastpath = sum(
            1 for idx in fast_indices if get_answer(data[idx], args) is True
        )
    else:
        num_neutral_in_fastpath = num_fast_path

    num_unneutral_in_fastpath = num_fast_path - num_neutral_in_fastpath
    fast_ratio = (num_fast_path / total) if total > 0 else 0.0
    adaptive_ratio = (num_adaptive_path / total) if total > 0 else 0.0
    fast_purity = (num_neutral_in_fastpath / num_fast_path) if num_fast_path > 0 else 0.0
    persona_hit_rate = (persona_hits / num_adaptive_path) if num_adaptive_path > 0 else 0.0

    # Composite score: Balances Persona Hit Rate (quality) and Fast Path Ratio (efficiency)
    # Penalizes low purity in Fast Path
    score = persona_hit_rate * (0.5 + 0.5 * fast_purity) * (1.0 + 0.5 * fast_ratio)

    return {
        "top_c": top_c,
        "top_k": top_k,
        "num_fast": num_fast_path,
        "fast_ratio": fast_ratio,
        "num_adapt": num_adaptive_path,
        "adapt_ratio": adaptive_ratio,
        "neutral_in_fast": num_neutral_in_fastpath,
        "unneutral_in_fast": num_unneutral_in_fastpath,
        "fast_purity": fast_purity,
        "persona_hits": persona_hits,
        "persona_hit_rate": persona_hit_rate,
        "composite_score": score,
    }


def grid_search(args):
    """Perform grid search over top_c_list and top_k_list."""
    random.seed(args.random_seed)
    data = load_dataset(args)
    if not data:
        print("No data to process.")
        return

    # Initialize Cross-Encoder model
    print(f"\nInitializing Cross-Encoder: {args.model_name} on {args.device}...")
    router = CrossEncoderRouter(args.model_name, args.device, args.dataset)

    # Load cache if available
    if args.ce_cache_file and os.path.exists(args.ce_cache_file):
        load_ce_cache(router, args.ce_cache_file)
        print(f"Loaded {len(router._cache)} cached scores from {args.ce_cache_file}")

    # Precompute / retrieve scores for all premises (run model inference once)
    print("\nExtracting Cross-Encoder scores for all samples...")
    premises = [f"{line.get('context', '')} {line.get('question', '')}" for line in data]
    all_scores = []
    for p in tqdm(premises, desc="Scoring premises"):
        all_scores.append(router.score(p))

    # Save cache if requested
    if args.ce_cache_file:
        os.makedirs(Path(args.ce_cache_file).parent, exist_ok=True)
        save_ce_cache(router, args.ce_cache_file)
        print(f"Saved {len(router._cache)} scores to cache {args.ce_cache_file}")

    # Run Grid Search across all (top_c, top_k) combinations
    print(f"\nSearching over top_c={args.top_c_list} and top_k={args.top_k_list}...")
    records = []
    for c in args.top_c_list:
        for k in args.top_k_list:
            res = compute_metrics_for_kc(data, all_scores, args, top_c=c, top_k=k)
            records.append(res)

    records = sorted(records, key=lambda x: x["composite_score"], reverse=True)

    # Print nicely formatted summary table
    print("\n" + "=" * 105)
    print(f"GRID SEARCH RESULTS FOR DATASET: {args.dataset} (Total samples: {len(data)})")
    print("=" * 105)
    header = (
        f"{'top_c':<7} {'top_k':<7} {'Fast Path':<12} {'Fast Purity':<14} "
        f"{'Adapt Path':<12} {'Persona Hit Rate':<18} {'Score':<8}"
    )
    print(header)
    print("-" * 105)

    for row in records:
        c = int(row["top_c"])
        k = int(row["top_k"])
        fp = f"{int(row['num_fast'])} ({row['fast_ratio']*100:.1f}%)"
        purity = f"{row['fast_purity']*100:.1f}%"
        ap = f"{int(row['num_adapt'])} ({row['adapt_ratio']*100:.1f}%)"
        hit_rate = f"{row['persona_hit_rate']*100:.1f}% ({int(row['persona_hits'])})"
        sc = f"{row['composite_score']:.4f}"
        print(f"{c:<7} {k:<7} {fp:<12} {purity:<14} {ap:<12} {hit_rate:<18} {sc:<8}")

    print("=" * 105)

    best_row = records[0]
    print(f"\n>>> RECOMMENDED BEST CONFIGURATION:")
    print(f"    --top_c {int(best_row['top_c'])}")
    print(f"    --top_k {int(best_row['top_k'])}")
    print(f"    Fast Path Ratio    : {best_row['fast_ratio']*100:.1f}%")
    print(f"    Fast Path Purity   : {best_row['fast_purity']*100:.1f}%")
    print(f"    Persona Hit Rate   : {best_row['persona_hit_rate']*100:.1f}%")
    print(f"    Composite Score    : {best_row['composite_score']:.4f}\n")

    # Save to CSV
    if args.output_csv:
        os.makedirs(Path(args.output_csv).parent, exist_ok=True)
        if pd is not None:
            df = pd.DataFrame(records)
            df.to_csv(args.output_csv, index=False)
        else:
            import csv
            with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        print(f"Full results saved to {args.output_csv}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Grid search best (k, c) parameters for Cross-Encoder routing")

    parser.add_argument(
        "--top_c_list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of top_c values to evaluate (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--top_k_list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of top_k values to evaluate (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BBQ",
        choices=["BBQ", "stereoset"],
        help="Dataset name (BBQ or stereoset)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing dataset folders (default: data)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-encoder/nli-distilroberta-base",
        help="Cross-Encoder model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to run Cross-Encoder (cuda, cpu)",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Whether truncate the dataset (same as MPT: 20 groups of 4 for BBQ, 100 for StereoSet)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for sampling reproducibility (default: 0)",
    )
    parser.add_argument(
        "--ce_cache_file",
        type=str,
        default="predictions/ce_cache.jsonl",
        help="Path to Cross-Encoder score cache file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/grid_search_k_c.csv",
        help="Path to save grid search result CSV",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    grid_search(args)
