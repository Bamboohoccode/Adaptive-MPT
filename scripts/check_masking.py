#!/usr/bin/env python3
"""Audit entity masking quality per dataset and category.

Reports, for every category:
  - unknown_kept   : the undetermined option survives masking unchanged (must be 100%)
  - ctx_masked     : the context was changed by masking
  - identity_leak  : an original identity string is still present after masking
                     (BBQ: answer names; StereoSet: the target word)
  - gender_cue     : StereoSet gender only - a gendered word is still present

Usage: python scripts/check_masking.py [--dataset BBQ stereoset BiasFreeBench] [--examples N]
"""
import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load data.py on its own so this check runs without vLLM/torch (the package __init__ imports them)
_spec = importlib.util.spec_from_file_location("adaptive_mpt_data", ROOT / "adaptive_mpt" / "data.py")
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
set_tasks_BBQ = _data.set_tasks_BBQ
set_tasks_stereoset = _data.set_tasks_stereoset
set_tasks_BiasFreeBench = _data.set_tasks_BiasFreeBench
mask_entities = _data.mask_entities
get_undetermined_index = _data.get_undetermined_index
GENDERED_WORDS = _data.GENDERED_WORDS

GENDER_CUES = re.compile(
    r"\b(he|she|him|his|her|hers|male|female|" + "|".join(map(re.escape, GENDERED_WORDS)) + r")\b",
    re.IGNORECASE,
)

TASKS = {
    "BBQ": set_tasks_BBQ,
    "stereoset": set_tasks_stereoset,
    "BiasFreeBench": set_tasks_BiasFreeBench,
}

def contains(text, phrase):
    return bool(phrase) and re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None

def identity_strings(item):
    """Original identity strings that masking is supposed to remove."""
    if "answer_info" in item:
        names = []
        for key in ("ans0", "ans1", "ans2"):
            name, label = item["answer_info"][key][0], item["answer_info"][key][1]
            if label != "unknown" and name.strip():
                names.append(name.strip())
        return names
    return [str(item.get("target", "")).strip()]

def audit(dataset, n_examples):
    rows = []
    for task in TASKS[dataset]():
        records = [json.loads(line) for line in open(ROOT / "data" / dataset / f"{task}.jsonl", encoding="utf-8")]
        stats = dict(n=0, unknown_kept=0, ctx_masked=0, identity_leak=0, gender_cue=0)
        leaks = []
        for item in records:
            masked = mask_entities(item)
            u = get_undetermined_index(item)
            stats["n"] += 1
            stats["unknown_kept"] += masked[f"ans{u}"] == item[f"ans{u}"]
            stats["ctx_masked"] += masked["context"] != item["context"]

            text = " ".join([masked["context"], masked["question"]] + [masked[f"ans{i}"] for i in range(3) if i != u])
            # Drop placeholders so a name like "person" does not match inside "[Person A]"
            text = re.sub(r"\[[A-Za-z]+ [A-Z]\]", " ", text)
            # Check both the annotated strings and the parts the masker split them into
            # (e.g. "Juan Rodriguez physician" -> "Juan Rodriguez" + "physician")
            candidates = set(identity_strings(item)) | set(masked["_entity_map"])
            leaked = sorted(s for s in candidates if contains(text, s))
            if leaked:
                stats["identity_leak"] += 1
                leaks.append((leaked, masked["context"]))
            if dataset == "stereoset" and str(item.get("category", "")).lower() == "gender" and GENDER_CUES.search(text):
                stats["gender_cue"] += 1
        rows.append((task, stats))
        for leaked, ctx in random.sample(leaks, min(n_examples, len(leaks))):
            print(f"  [{task}] leak {leaked}: {ctx[:160]}")
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["BBQ", "stereoset"], choices=list(TASKS))
    parser.add_argument("--examples", type=int, default=0, help="Print N leaking examples per category")
    args = parser.parse_args()
    random.seed(0)

    for dataset in args.dataset:
        rows = audit(dataset, args.examples)
        print(f"\n## {dataset}")
        header = f"{'category':22s} {'n':>6s} {'unknown_kept':>13s} {'ctx_masked':>11s} {'identity_leak':>14s}"
        if dataset == "stereoset":
            header += f" {'gender_cue':>11s}"
        print(header)
        total = dict(n=0, unknown_kept=0, ctx_masked=0, identity_leak=0, gender_cue=0)
        for task, s in rows + [("TOTAL", None)]:
            if s is None:
                s = total
            else:
                for k in total:
                    total[k] += s[k]
            pct = lambda k: f"{100 * s[k] / s['n']:.1f}%"
            line = f"{task:22s} {s['n']:6d} {pct('unknown_kept'):>13s} {pct('ctx_masked'):>11s} {pct('identity_leak'):>14s}"
            if dataset == "stereoset":
                line += f" {pct('gender_cue'):>11s}"
            print(line)

if __name__ == "__main__":
    main()
