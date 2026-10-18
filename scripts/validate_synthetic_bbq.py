"""Validate synthetic BBQ scenarios/rows (schema, entity grounding, overlap with real BBQ).

Usage:
    python scripts/validate_synthetic_bbq.py [--dir data/synthetic_bbq] [--bbq_dir data/BBQ]
"""
import argparse
import json
import re
from pathlib import Path

REQUIRED_SCENARIO_KEYS = (
    "entity0", "entity1", "ans0", "ans1", "group0", "group1", "stereotyped_group",
    "ambig_context", "disambig_addition", "neg_question", "nonneg_question", "neg_answer",
)
REQUIRED_ROW_KEYS = (
    "example_id", "question_index", "question_polarity", "context_condition", "category",
    "answer_info", "additional_metadata", "context", "question", "ans0", "ans1", "ans2", "label",
)
UNKNOWN_ANSWER = "Can't be determined"
BBQ_OVERLAP_THRESHOLD = 0.5


def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def ngrams(text, n=6):
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def load_bbq_ngrams(bbq_dir, category):
    """6-gram set over all real BBQ contexts of this category (empty if file missing)."""
    path = Path(bbq_dir) / f"{category}.jsonl"
    grams = set()
    if not path.exists():
        return grams
    with open(path) as f:
        for line in f:
            grams |= ngrams(json.loads(line)["context"])
    return grams


def validate_scenario(s, bbq_grams=None):
    """Return a list of problems (empty list = scenario is valid)."""
    if not isinstance(s, dict):
        return ["scenario is not an object"]
    problems = [f"missing key: {k}" for k in REQUIRED_SCENARIO_KEYS if k not in s]
    if problems:
        return problems
    for k in REQUIRED_SCENARIO_KEYS:
        if k != "neg_answer" and not (isinstance(s[k], str) and s[k].strip()):
            problems.append(f"empty or non-string: {k}")
    if s["neg_answer"] not in (0, 1):
        problems.append("neg_answer must be 0 or 1")
    if problems:
        return problems

    amb = s["ambig_context"].lower()
    for i in (0, 1):
        if s[f"entity{i}"].lower() not in amb:
            problems.append(f"entity{i} not verbatim in ambig_context")
        if s[f"entity{i}"].lower() not in s[f"ans{i}"].lower():
            problems.append(f"ans{i} does not contain entity{i}")
    if s["entity0"].lower() == s["entity1"].lower():
        problems.append("entity0 == entity1")
    if s["group0"] == s["group1"]:
        problems.append("group0 == group1")
    if s["stereotyped_group"] not in (s["group0"], s["group1"]):
        problems.append("stereotyped_group must equal group0 or group1")
    if not s["neg_question"].strip().endswith("?") or not s["nonneg_question"].strip().endswith("?"):
        problems.append("questions must end with '?'")
    if s["neg_question"].strip().lower() == s["nonneg_question"].strip().lower():
        problems.append("neg_question == nonneg_question")
    if bbq_grams:
        grams = ngrams(s["ambig_context"] + " " + s["disambig_addition"])
        if grams and len(grams & bbq_grams) / len(grams) > BBQ_OVERLAP_THRESHOLD:
            problems.append("too similar to a real BBQ context (6-gram overlap)")
    return problems


def scenario_to_rows(s, category, scenario_idx, start_id, split):
    """Expand one scenario into the 4 BBQ rows: neg/ambig, neg/disambig, nonneg/ambig, nonneg/disambig."""
    answer_info = {
        "ans0": [s["entity0"].lower(), s["group0"]],
        "ans1": [s["entity1"].lower(), s["group1"]],
        "ans2": [UNKNOWN_ANSWER, "unknown"],
    }
    meta = {"subcategory": "None", "stereotyped_groups": [s["stereotyped_group"]],
            "version": "a", "source": "synthetic"}
    disambig = f'{s["ambig_context"].strip()} {s["disambig_addition"].strip()}'
    neg_ans = int(s["neg_answer"])
    spec = [
        ("neg", "ambig", s["ambig_context"].strip(), s["neg_question"], 2),
        ("neg", "disambig", disambig, s["neg_question"], neg_ans),
        ("nonneg", "ambig", s["ambig_context"].strip(), s["nonneg_question"], 2),
        ("nonneg", "disambig", disambig, s["nonneg_question"], 1 - neg_ans),
    ]
    rows = []
    for j, (pol, cond, ctx, q, label) in enumerate(spec):
        rows.append({
            "example_id": start_id + j, "question_index": str(scenario_idx + 1),
            "question_polarity": pol, "context_condition": cond, "category": category,
            "answer_info": answer_info, "additional_metadata": meta,
            "context": ctx, "question": q.strip(),
            "ans0": s["ans0"].strip(), "ans1": s["ans1"].strip(), "ans2": UNKNOWN_ANSWER,
            "label": label, "split": split,
        })
    return rows


def validate_rows(rows):
    """Row-level checks on a written file. Returns list of problems."""
    problems = []
    if len(rows) % 4:
        problems.append(f"row count {len(rows)} not a multiple of 4")
    for r in rows:
        miss = [k for k in REQUIRED_ROW_KEYS if k not in r]
        if miss:
            problems.append(f"example {r.get('example_id')}: missing {miss}")
            continue
        if r["label"] not in (0, 1, 2):
            problems.append(f"example {r['example_id']}: bad label")
        if r["context_condition"] == "ambig" and r["label"] != 2:
            problems.append(f"example {r['example_id']}: ambig must have label 2")
        if r["context_condition"] == "disambig" and r["label"] == 2:
            problems.append(f"example {r['example_id']}: disambig must not have label 2")
    for g in range(0, len(rows) - len(rows) % 4, 4):
        grp = rows[g:g + 4]
        if grp[1]["label"] + grp[3]["label"] != 1:
            problems.append(f"group at {g}: neg/nonneg disambig answers are not complementary")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/synthetic_bbq")
    ap.add_argument("--bbq_dir", default="data/BBQ")
    args = ap.parse_args()
    total, bad = 0, 0
    for path in sorted(Path(args.dir).glob("*.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        probs = validate_rows(rows)
        total += len(rows)
        bad += len(probs)
        by_split = {}
        for r in rows:
            by_split[r.get("split")] = by_split.get(r.get("split"), 0) + 1
        print(f"{path.name}: {len(rows)} rows {by_split} -> {'OK' if not probs else probs}")
    print(f"TOTAL rows={total} problems={bad}")


if __name__ == "__main__":
    main()
