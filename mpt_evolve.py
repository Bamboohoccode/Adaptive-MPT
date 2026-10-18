#!/usr/bin/env python3
"""Evolve Adaptive-MPT prompts on Synthetic-BBQ (see adaptive_mpt/evolve.py).

    python scripts/gen_synthetic_bbq.py                       # 1. build data/synthetic_bbq/
    python mpt_evolve.py --models Qwen/Qwen2.5-7B-Instruct    # 2. evolve (resumable with --resume)
    python mpt_evolve.py --apply evolved/<run>/best_prompts.json   # 3. overwrite adaptive_mpt/prompts.py (backup kept)
"""
import argparse
import ast
import json
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")

import adaptive_mpt.prompts as prompts_module
from adaptive_mpt import evolve as ev


def parse_arguments():
    p = argparse.ArgumentParser(description="Evolutionary prompt optimisation on Synthetic-BBQ")
    # data / output
    p.add_argument("--data_dir", default="data/synthetic_bbq")
    p.add_argument("--out_dir", default="evolved/run1")
    p.add_argument("--categories", nargs="*", default=None, help="restrict to some categories")
    # evolution
    p.add_argument("--genes", nargs="+", default=list(ev.DEFAULT_GENES),
                   help=f"prompt constants to evolve; also possible: {', '.join(ev.OPTIONAL_GENES)}")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--mut_prob", type=float, default=0.5, help="probability that a gene is mutated in a round")
    p.add_argument("--mutation_mode", choices=["word", "token"], default="word",
                   help="word: swap an unprotected word with a prompt-lexicon word; token: tokenizer-level swap")
    p.add_argument("--tokenizer", default=None, help="tokenizer for --mutation_mode token (default: first --models)")
    p.add_argument("--lam", type=float, default=1.0, help="fitness = acc_avg - lam * diff_bias_avg")
    p.add_argument("--margin", type=float, default=0.005, help="child must beat the other individual by more than this")
    p.add_argument("--screen_groups", type=int, default=8, help="4-row groups used for the cheap screening pass")
    p.add_argument("--screen_margin", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--apply", default=None, metavar="BEST_PROMPTS_JSON",
                   help="only overwrite adaptive_mpt/prompts.py with this file (backup -> prompts.py.bak), then exit")
    p.add_argument("--no_apply", action="store_true",
                   help="do not overwrite prompts.py at the end of the run (default: overwrite, with backup)")
    p.add_argument("--force_apply", action="store_true",
                   help="overwrite prompts.py even if the evolved prompts are not better than the originals on the Select set")
    # model / engine (same names as mpt_inference.py)
    p.add_argument("--models", nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"])
    p.add_argument("--inference_type", default="vllm", choices=["vllm", "openai"])
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--chat_templates", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument("--num_rounds", type=int, default=2, help="tribunal debate rounds")
    p.add_argument("--cot_guard", action="store_true", default=False)
    args = p.parse_args()

    # fixed pipeline settings (identical to mpt_inference.py)
    args.dataset = "BBQ"
    args.method = "evolve"
    args.overwrite = True
    args.router_method = "context_sufficiency"
    args.fast_path_method = "logical"
    args.debate_mode = "asymmetric_tribunal"
    args.asymmetric = True
    args.summary_id = 0
    args.truncate = False
    args.data_split = None
    return args


def apply_best_prompts(best_json, prompts_path="adaptive_mpt/prompts.py"):
    """Rewrite the assignment of each evolved constant in prompts.py (keeps every other line)."""
    best = json.loads(Path(best_json).read_text())
    src = Path(prompts_path).read_text()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    spans = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in best and ast.literal_eval(node.value) != best[name]:  # skip unchanged constants
                spans[name] = (node.lineno - 1, node.end_lineno)
    defined = {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    missing = set(best) - defined
    if missing:
        raise SystemExit(f"constants not found in {prompts_path}: {sorted(missing)}")
    if not spans:
        print("No constant differs from the current prompts: nothing to write.")
        return
    shutil.copy(prompts_path, str(prompts_path) + ".bak")
    for name, (start, end) in sorted(spans.items(), key=lambda kv: -kv[1][0]):
        body = best[name].replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        lines[start:end] = [f'{name} = """{body}"""\n']
    Path(prompts_path).write_text("".join(lines))
    print(f"Updated {sorted(spans)} in {prompts_path} (backup: {prompts_path}.bak)")


def main():
    args = parse_arguments()
    if args.apply:
        apply_best_prompts(args.apply)
        return

    import numpy as np
    import torch
    from adaptive_mpt.models import LLMDecoder

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    unknown = [g for g in args.genes if not hasattr(prompts_module, g)]
    if unknown:
        raise SystemExit(f"unknown genes: {unknown}")

    evolve_records = ev.load_synthetic_records(args.data_dir, "evolve", args.categories)
    select_records = ev.load_synthetic_records(args.data_dir, "select", args.categories)
    print(f"Evolve set: {len(evolve_records)} rows | Select set: {len(select_records)} rows")

    originals = [getattr(prompts_module, g) for g in args.genes]
    if args.mutation_mode == "token":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.models[0])
        vocab_words = ev.vocab_words_from_tokenizer(tokenizer)
        mutate_fn = lambda text, orig: ev.mutate(text, orig, rng, "token", tokenizer=tokenizer, vocab_words=vocab_words)  # noqa: E731
    else:
        lexicon = ev.build_lexicon(originals)
        mutate_fn = lambda text, orig: ev.mutate(text, orig, rng, "word", lexicon=lexicon)  # noqa: E731

    state = ev.EvolutionState(out_dir)
    if args.resume and not state.load():
        print("--resume given but no state.json found: starting fresh")
    rng = random.Random(args.seed * 1000 + state.data["round"])

    models = [LLMDecoder(args, m) for m in args.models]
    evaluator = ev.Evaluator(args, models, cache=state.data["cache"], lam=args.lam)

    state = ev.evolve(args, evaluator, evolve_records, out_dir, args.genes, rng, mutate_fn, state)

    # Final: pick the best individual per gene and compare with the original prompts on the held-out Select set.
    best = ev.active_overrides(state.data["genes"])
    baseline = {g: getattr(prompts_module, g) for g in args.genes}
    m_base = evaluator.metrics(baseline, select_records, "select")
    m_best = evaluator.metrics(best, select_records, "select")
    ev.plot_select_summary(m_base, m_best, out_dir / "plots")
    (out_dir / "best_prompts.json").write_text(json.dumps(best, ensure_ascii=False, indent=2))
    (out_dir / "select_metrics.json").write_text(json.dumps({"baseline": m_base, "evolved": m_best}, indent=2))
    print("Select set  baseline:", {k: round(v, 3) for k, v in m_base.items() if k.startswith(("acc", "diff"))})
    print("Select set  evolved :", {k: round(v, 3) for k, v in m_best.items() if k.startswith(("acc", "diff"))})
    print(f"Saved {out_dir / 'best_prompts.json'}")

    # Overwrite prompts.py, but only with prompts that are not worse than the originals on the held-out Select set.
    f_base, f_best = ev.fitness_of(m_base, args.lam), ev.fitness_of(m_best, args.lam)
    if args.no_apply:
        print(f"--no_apply: prompts.py untouched. Later: python mpt_evolve.py --apply {out_dir / 'best_prompts.json'}")
    elif f_best >= f_base or args.force_apply:
        changed = {g: t for g, t in best.items() if t != baseline[g]}
        if changed:
            apply_best_prompts(out_dir / "best_prompts.json")
        else:
            print("Evolved prompts equal the originals: prompts.py untouched.")
    else:
        print(f"Select fitness evolved {f_best:.4f} < baseline {f_base:.4f}: prompts.py NOT overwritten "
              f"(use --force_apply to override; best prompts are in {out_dir / 'best_prompts.json'})")


if __name__ == "__main__":
    main()
