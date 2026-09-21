#!/usr/bin/env python3
"""
Adaptive-MPT: Efficient & Debiased LLM Reasoning via Counterfactual Dual-Pass Screening and Asymmetric Persona Tribunal.

Router is strictly fixed to Zero-Oracle 'context_sufficiency'.
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from adaptive_mpt.data import set_tasks_BBQ, set_tasks_stereoset, load_dataset_records, save_dataset_records
from adaptive_mpt.models import LLMDecoder, get_total_llm_calls
from adaptive_mpt.pipeline import (
    run_dual_pass_screening,
    route_samples,
    execute_fast_path,
    execute_asymmetric_tribunal,
)

# Backward-compatibility alias
Decoder = LLMDecoder

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive-MPT: Dual-Pass Context Sufficiency Screening & Asymmetric Tribunal")

    # Benchmark & Model
    parser.add_argument("--models", type=str, nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"], help="Model name or path")
    parser.add_argument("--dataset", type=str, default="BBQ", choices=["BBQ", "stereoset"], help="Benchmark dataset")
    parser.add_argument("--truncate", action="store_true", help="Sample standard 880-sample evaluation subset on BBQ")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for deterministic sampling")

    # Hardware & Engine
    parser.add_argument("--inference_type", type=str, default="vllm", choices=["vllm", "openai"], help="Inference backend")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory fraction")
    parser.add_argument("--max_model_len", type=int, default=32768, help="vLLM max context length")
    parser.add_argument("--chat_templates", action="store_true", help="Apply official tokenizer chat template")

    # Generation Parameters
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=768, help="Max output tokens")
    parser.add_argument("--num_rounds", type=int, default=2, help="Tribunal debate rounds")

    # Run Metadata
    parser.add_argument("--method", type=str, default="mpt", help="Method tag for output path")
    parser.add_argument("--note", type=str, default="1", help="Experiment note subfolder")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    parser.add_argument("--split", action="store_true", help="Process tasks sequentially")

    # Framework Parameters (router_method is locked to context_sufficiency)
    parser.add_argument("--router_method", type=str, default="context_sufficiency", choices=["context_sufficiency"],
                        help="Routing method (fixed to context_sufficiency)")
    parser.add_argument("--fast_path_method", type=str, default="logical", help="Fast path strategy")
    parser.add_argument("--asymmetric", action="store_true", default=True, help="Asymmetric Tribunal mode (fixed)")
    parser.add_argument("--debate_mode", type=str, default="asymmetric_tribunal", help="Debate mode (fixed)")
    parser.add_argument("--cot_guard", "--over_refusal_guard", action="store_true", default=False,
                        help="Optional CoT keyword scanning to rescue potential over-refusal in Pass 2")

    args = parser.parse_args()

    # Route method is strictly context_sufficiency
    args.router_method = "context_sufficiency"
    args.debate_mode = "asymmetric_tribunal"
    args.asymmetric = True
    args.summary_id = 0

    if args.dataset == "BBQ":
        tasks = set_tasks_BBQ()
        args.input_paths = [f"./data/{args.dataset}/{task}.jsonl" for task in tasks]
    elif args.dataset == "stereoset":
        tasks = set_tasks_stereoset()
        args.input_paths = [f"./data/{args.dataset}/{task}.jsonl" for task in tasks]

    return args

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    experiment_signature = f"{args.method}_{args.num_rounds}_{args.max_tokens}"

    print("=" * 80)
    print("Adaptive-MPT Pipeline: Router = context_sufficiency (Fixed) | Mode = Asymmetric Tribunal")
    print(f"Model: {args.models[0]} | Dataset: {args.dataset} | Truncate: {args.truncate} | Seed: {args.random_seed}")
    print("=" * 80)

    models = [LLMDecoder(args, model) for model in args.models]
    path_groups = [[p] for p in args.input_paths] if args.split else [args.input_paths]

    for batch_input_paths in path_groups:
        sample_records, category_manifest = load_dataset_records(args, batch_input_paths)
        print(f"\nCurrent Batch Total Samples: {len(sample_records)}")

        # 1. Dual-Pass Counterfactual Screening (Batch 2N)
        p_masked, p_unmasked, cot_unmasked, fast_path_cache = run_dual_pass_screening(args, models, sample_records)

        # 2. Zero-Oracle Consistency Gate (Fixed context_sufficiency)
        fast_path_indices, tribunal_indices = route_samples(args, sample_records, p_masked, p_unmasked, cot_unmasked)

        # 3. Fast Path (System 1: 0 Compute Overhead)
        execute_fast_path(args, models, fast_path_indices, sample_records, fast_path_cache, experiment_signature)

        # 4. Asymmetric Persona Tribunal (System 2: Fact-Anchor, Bias-Auditor, Judge)
        execute_asymmetric_tribunal(args, models, tribunal_indices, sample_records, experiment_signature)

        # 5. Save predictions
        save_dataset_records(sample_records, category_manifest)

    print(f"\nInference Complete. Total LLM Calls: {get_total_llm_calls()}\n")

if __name__ == "__main__":
    main()
