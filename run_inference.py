#!/usr/bin/env python3
"""
Adaptive-MPT: Asymmetric Persona Tribunal & Counterfactual Dual-Pass Screening.

Primary inference entry point.
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from adaptive_mpt.data import set_tasks_BBQ, set_tasks_stereoset, load_dataset_records, save_dataset_records
from adaptive_mpt.models import LLMDecoder, TOTAL_LLM_CALLS
from adaptive_mpt.pipeline import (
    run_dual_pass_screening,
    route_samples,
    execute_fast_path,
    execute_asymmetric_tribunal,
)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adaptive-MPT Inference")

    parser.add_argument("--method", type=str, default="mpt", help="Method identifier")
    parser.add_argument("--note", type=str, default="1", help="Experiment note/directory")
    parser.add_argument("--dataset", type=str, default="BBQ", choices=["BBQ", "stereoset"], help="Benchmark dataset")
    parser.add_argument("--models_dir", type=str, default=None, help="Local model weights directory")
    parser.add_argument("--inference_type", type=str, default="vllm", choices=["vllm", "openai"], help="Inference engine")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of debate agents")
    parser.add_argument("--models", type=str, nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"], help="Model names")
    parser.add_argument("--summary_id", type=int, default=0, help="Model ID used for judge summary")
    parser.add_argument("--num_rounds", type=int, default=2, help="Debate rounds")
    parser.add_argument("--num_attempts", type=int, default=5, help="Self-consistency attempts")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=768, help="Max new tokens")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=32768, help="vLLM max model context length")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions")
    parser.add_argument("--split", action="store_true", help="Process tasks sequentially")
    parser.add_argument("--no_general", action="store_true", help="Disable neutral general public persona")
    parser.add_argument("--truncate", action="store_true", help="Truncate dataset to standard evaluation subset")
    parser.add_argument("--chat_templates", action="store_true", help="Apply model chat template")

    # Adaptive-MPT Specific Flags
    parser.add_argument("--debate_mode", type=str, default="asymmetric_tribunal", choices=["symmetric", "asymmetric_tribunal"], help="Debate mode")
    parser.add_argument("--asymmetric", action="store_true", default=None, help="Enable Asymmetric Tribunal mode")
    parser.add_argument("--router_method", type=str, default="context_sufficiency", help="Router method")
    parser.add_argument("--fast_path_method", type=str, default="logical", choices=["direct", "logical", "logical_reprompting"], help="Fast path strategy")
    parser.add_argument("--feature_extraction", "--fact_extraction", action="store_true", help="Enable Quote-First evidence grounding")
    parser.add_argument("--disable_cross_encoder", action="store_true", help="Disable cross-encoder routing")
    parser.add_argument("--mask_entities", type=str, default="none", help="Entity masking mode")
    parser.add_argument("--cross_encoder_model", type=str, default="cross-encoder/nli-distilroberta-base")
    parser.add_argument("--cross_encoder_device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--top_c", type=int, default=2)
    parser.add_argument("--ce_cache_file", type=str, default=None)

    args = parser.parse_args()
    args.fact_extraction = args.feature_extraction

    if args.asymmetric is not None:
        args.debate_mode = "asymmetric_tribunal" if args.asymmetric else "symmetric"
    elif args.debate_mode == "asymmetric_tribunal":
        args.asymmetric = True
    else:
        args.asymmetric = False

    if args.method.endswith("no_general"):
        args.no_general = True

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
    print(f"Adaptive-MPT Pipeline Initialized: {experiment_signature}")
    print(f"Dataset: {args.dataset} | Truncate: {args.truncate} | Random Seed: {args.random_seed}")

    models = [LLMDecoder(args, model) for model in args.models]
    path_groups = [[p] for p in args.input_paths] if args.split else [args.input_paths]

    for batch_input_paths in path_groups:
        sample_records, category_manifest = load_dataset_records(args, batch_input_paths)
        print(f"\nCurrent Batch Total Samples: {len(sample_records)}")

        # 1. Dual-Pass Counterfactual Screening (Batch 2N)
        p_masked, p_unmasked, cot_unmasked, fast_path_cache = run_dual_pass_screening(args, models, sample_records)

        # 2. Zero-Oracle Consistency Gate
        fast_path_indices, tribunal_indices = route_samples(args, sample_records, p_masked, p_unmasked, cot_unmasked)

        # 3. Fast Path (System 1)
        execute_fast_path(args, models, fast_path_indices, sample_records, fast_path_cache, experiment_signature)

        # 4. Asymmetric Persona Tribunal (System 2)
        execute_asymmetric_tribunal(args, models, tribunal_indices, sample_records, experiment_signature)

        # 5. Save predictions
        save_dataset_records(sample_records, category_manifest)

    print(f"\nInference Complete. Total LLM Calls: {TOTAL_LLM_CALLS}\n")

if __name__ == "__main__":
    main()
