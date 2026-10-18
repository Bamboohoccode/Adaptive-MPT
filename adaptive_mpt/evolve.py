"""Evolutionary prompt optimisation on Synthetic-BBQ.

Each gene (one prompt constant of adaptive_mpt.prompts) holds two individuals (A, B). Every round, for every gene:
    parent P (random of A/B) --token/word mutation with prob p--> child C
    C is compared with the other individual Q on the Evolve set (fitness = acc_avg - lambda * diff_bias_avg)
    C replaces Q if it is better by more than `margin`.
The pipeline in adaptive_mpt/pipeline.py is left untouched: candidate prompts are injected by setting the
module-level names of `adaptive_mpt.pipeline` just for the duration of an evaluation.
"""
import copy
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

import adaptive_mpt.prompts as prompts_module

# Genes that are actually used by the live pipeline (see pipeline.py); the rest of prompts.py is dead code.
DEFAULT_GENES = (
    "LOGICAL_PROMPT_MASKED",
    "LOGICAL_PROMPT_ASYMMETRIC_UNMASKED",
    "REVIEW_PROMPT",
    "SUMMARY_PROMPT",
)
OPTIONAL_GENES = (
    "DEFAULT_SYS_PROMPT",
    "FACT_ANCHOR_SYS_PROMPT",
    "BIAS_AUDITOR_SYS_PROMPT",
    "JUDGE_PERSONA_SYS_PROMPT",
)

# Spans that a mutation may never touch (placeholders, answer contract, undetermined option, role tags).
PROTECTED_PATTERNS = (
    r"\{[A-Za-z_]+\}",
    r"<answer>\s*\d?\s*</answer>",
    r"</?answer>",
    r"Cannot be determined",
    r"\[[A-Za-z\- ]+\]",
    r"\b(?:Fact-Anchor|Bias-Auditor|Judicial Arbiter)\b",
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']*")


# ── Mutation ──────────────────────────────────────────────────────────────────
# output: [1,3,7,9] 1->3 is protected, 7->9 also
def protected_spans(text):
    spans = []
    for pat in PROTECTED_PATTERNS:
        spans.extend(m.span() for m in re.finditer(pat, text))
    return spans


def _overlaps(span, spans):
    return any(span[0] < e and s < span[1] for s, e in spans)


def build_lexicon(texts, min_len=3):
    """Words occurring in the prompts, used as replacement candidates (keeps mutations on-topic)."""
    words = set()
    for t in texts:
        spans = protected_spans(t)
        for m in WORD_RE.finditer(t):
            if len(m.group()) >= min_len and not _overlaps(m.span(), spans):
                words.add(m.group().lower())
    return sorted(words)


def mutate_word(text, rng, lexicon):
    """Replace one random unprotected word by a random lexicon word. Returns (new_text, description) or None."""
    spans = protected_spans(text)
    cands = [m for m in WORD_RE.finditer(text) if len(m.group()) >= 3 and not _overlaps(m.span(), spans)]
    if not cands:
        return None
    m = cands[rng.randrange(len(cands))]
    old = m.group()
    pool = [w for w in lexicon if w != old.lower()]
    if not pool:
        return None
    new = rng.choice(pool)
    if old[0].isupper():
        new = new.capitalize()
    return text[:m.start()] + new + text[m.end():], f"{old}->{new}"


def mutate_token(text, rng, tokenizer, vocab_words):
    """Tokenise, `randint` one token that is a whole unprotected word, replace it by another vocab token
    of the same kind, decode back to text. Returns (new_text, description) or None."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = protected_spans(text)
    cands = []
    for tok_id, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        piece = text[s:e].strip()
        if len(piece) >= 3 and WORD_RE.fullmatch(piece) and not _overlaps((s, e), spans):
            # whole word only (no sub-word fragment): neighbours must not be letters
            if (s == 0 or not text[s - 1].isalpha()) and (e >= len(text) or not text[e].isalpha()):
                cands.append((s, e, piece))
    if not cands:
        return None
    s, e, old = cands[rng.randrange(len(cands))]
    new = rng.choice([w for w in vocab_words if w.lower() != old.lower()])
    lead = " " if text[s:e].startswith(" ") else ""
    return text[:s] + lead + new + text[e:], f"{old}->{new}"


def vocab_words_from_tokenizer(tokenizer, max_words=20000):
    """Alphabetic, space-prefixed vocabulary entries (whole English-looking words) of the tokenizer."""
    words = []
    for tok in tokenizer.get_vocab():
        w = tok.lstrip("Ġ▁ ")
        if (tok != w) and w.isascii() and w.isalpha() and w.islower() and 3 <= len(w) <= 12:
            words.append(w)
    words = sorted(set(words))
    return words[:max_words] if len(words) > max_words else words


def validate_prompt(candidate, original):
    """True iff every protected span of the original survives (same multiset) and the length stays sane."""
    for pat in PROTECTED_PATTERNS:
        if sorted(re.findall(pat, original)) != sorted(re.findall(pat, candidate)):
            return False
    if candidate.count("{") != original.count("{") or candidate.count("}") != original.count("}"):
        return False
    return 0.5 * len(original) <= len(candidate) <= 1.5 * len(original)


def mutate(text, original, rng, mode, lexicon=None, tokenizer=None, vocab_words=None, max_tries=20):
    for _ in range(max_tries):
        out = mutate_token(text, rng, tokenizer, vocab_words) if mode == "token" else mutate_word(text, rng, lexicon)
        if out and out[0] != text and validate_prompt(out[0], original):
            return out
    return None


# ── Evaluation ────────────────────────────────────────────────────────────────

def load_synthetic_records(data_dir, split, categories=None):
    """Rows of data/synthetic_bbq/*.jsonl whose `split` field equals `split` ('evolve' | 'select')."""
    records = []
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        if categories and path.stem not in categories:
            continue
        for line in open(path):
            r = json.loads(line)
            if r.get("split") == split:
                r["_task_category"] = path.stem
                records.append(r)
    if len(records) % 4:
        raise ValueError(f"{split}: {len(records)} rows is not a multiple of 4 (groups were broken)")
    return records


def sample_groups(records, n_groups, seed):
    """Random subset of whole 4-row groups (used for the cheap screening pass)."""
    groups = [records[i:i + 4] for i in range(0, len(records), 4)]
    rng = random.Random(seed)
    rng.shuffle(groups)
    return [r for g in groups[:n_groups] for r in g]


def config_key(overrides, tag):
    blob = json.dumps(overrides, sort_keys=True) + "|" + tag
    return hashlib.sha1(blob.encode()).hexdigest()


def fitness_of(metrics, lam):
    return metrics["acc_avg"] - lam * metrics["diff_bias_avg"]


class Evaluator:
    """Runs the existing Adaptive-MPT pipeline with prompt overrides and returns BBQ metrics (cached)."""

    def __init__(self, args, models, cache=None, lam=1.0):
        self.args, self.models, self.lam = args, models, lam
        self.cache = cache if cache is not None else {}

    def metrics(self, overrides, records, tag):
        key = config_key(overrides, tag)
        if key not in self.cache:
            self.cache[key] = self._run(overrides, records)
        return self.cache[key]

    def _run(self, overrides, records):
        import mpt_evaluate
        from adaptive_mpt import pipeline

        sig = f"{self.args.method}_{self.args.num_rounds}_{self.args.max_tokens}"
        recs = copy.deepcopy(records)
        saved = {name: getattr(pipeline, name) for name in overrides}
        try:
            for name, text in overrides.items():
                setattr(pipeline, name, text)
            p_m, p_u, cot_u, cache = pipeline.run_dual_pass_screening(self.args, self.models, recs)
            fast, tribunal = pipeline.route_samples(self.args, recs, p_m, p_u, cot_u)
            pipeline.execute_fast_path(self.args, self.models, fast, recs, cache, sig)
            pipeline.execute_asymmetric_tribunal(self.args, self.models, tribunal, recs, sig)
        finally:
            for name, text in saved.items():
                setattr(pipeline, name, text)
        metrics = mpt_evaluate.compute_metrics(pd.DataFrame(recs), sig, "BBQ")
        return {k: float(v) for k, v in metrics.items()}


# ── Evolution loop ────────────────────────────────────────────────────────────

class EvolutionState:
    """Everything needed to resume a run: individuals per gene, round counter, metric cache.
    The RNG is re-seeded from (seed, round) on resume, so it is not stored."""

    def __init__(self, out_dir):
        self.path = Path(out_dir) / "state.json"
        self.data = {"round": 0, "genes": {}, "cache": {}}

    def load(self):
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            return True
        return False

    def save(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False))


def active_overrides(genes):
    """Currently best individual of each gene (A wins ties)."""
    return {g: (v["A"] if v["fit_A"] >= v["fit_B"] else v["B"]) for g, v in genes.items()}


def evolve(args, evaluator, evolve_records, out_dir, genes, rng, mutate_fn, state):
    """Run `args.generations` rounds. `mutate_fn(text, original) -> (new_text, desc) | None`."""
    out_dir = Path(out_dir)
    hist_path = out_dir / "history.csv"
    screen = sample_groups(evolve_records, args.screen_groups, seed=args.seed)
    originals = {g: getattr(prompts_module, g) for g in genes}

    if not state.data["genes"]:
        for g in genes:
            child = mutate_fn(originals[g], originals[g])
            state.data["genes"][g] = {"A": originals[g], "B": child[0] if child else originals[g],
                                      "fit_A": None, "fit_B": None, "desc_B": child[1] if child else "copy"}
        _score_initial(args, evaluator, evolve_records, state)
        state.save()

    for rnd in range(state.data["round"] + 1, args.generations + 1):
        rows = []
        for g in genes:
            ind = state.data["genes"][g]
            parent_key = rng.choice(["A", "B"])
            other_key = "B" if parent_key == "A" else "A"
            row = {"round": rnd, "gene": g, "parent": parent_key, "mutated": False, "replaced": False,
                   "desc": "", "fit_child": None, "fit_other": None}
            if rng.random() >= args.mut_prob:
                row["desc"] = "no mutation (p)"
                rows.append(_finish_row(row, state, evaluator, g))
                continue
            out = mutate_fn(ind[parent_key], originals[g])
            if out is None:
                row["desc"] = "no valid mutation"
                rows.append(_finish_row(row, state, evaluator, g))
                continue
            child_text, desc = out
            row.update(mutated=True, desc=desc)

            ctx = {k: v for k, v in active_overrides(state.data["genes"]).items() if k != g}
            q_over, c_over = {**ctx, g: ind[other_key]}, {**ctx, g: child_text}

            # screening pass on a small group subset, then full pass only for promising children
            q_s = evaluator.metrics(q_over, screen, "screen")
            c_s = evaluator.metrics(c_over, screen, "screen")
            if fitness_of(c_s, args.lam) < fitness_of(q_s, args.lam) - args.screen_margin:
                row["desc"] += " [rejected at screening]"
                row["fit_child"], row["fit_other"] = fitness_of(c_s, args.lam), fitness_of(q_s, args.lam)
                rows.append(_finish_row(row, state, evaluator, g))
                continue
            q_m = evaluator.metrics(q_over, evolve_records, "evolve")
            c_m = evaluator.metrics(c_over, evolve_records, "evolve")
            f_c, f_q = fitness_of(c_m, args.lam), fitness_of(q_m, args.lam)
            row["fit_child"], row["fit_other"] = f_c, f_q
            if f_c > f_q + args.margin:
                ind[other_key], ind[f"fit_{other_key}"] = child_text, f_c
                row["replaced"] = True
            else:
                ind[f"fit_{other_key}"] = f_q
            rows.append(_finish_row(row, state, evaluator, g, winner_metrics=c_m if row["replaced"] else q_m))

        state.data["round"] = rnd
        state.data["cache"] = evaluator.cache
        state.save()
        _append_history(hist_path, rows)
        plot_history(hist_path, out_dir / "plots")
        print(f"[round {rnd}/{args.generations}] " + ", ".join(
            f"{r['gene']}:{'REPLACED' if r['replaced'] else 'kept'}" for r in rows))
    return state


def _score_initial(args, evaluator, records, state):
    genes = state.data["genes"]
    base = {g: v["A"] for g, v in genes.items()}
    for g, v in genes.items():
        ctx = {k: x for k, x in base.items() if k != g}
        for key in ("A", "B"):
            m = evaluator.metrics({**ctx, g: v[key]}, records, "evolve")
            v[f"fit_{key}"] = fitness_of(m, args.lam)


def _finish_row(row, state, evaluator, gene, winner_metrics=None):
    ind = state.data["genes"][gene]
    row["fit_A"], row["fit_B"] = ind["fit_A"], ind["fit_B"]
    for k in ("acc_a", "acc_d", "acc_avg", "diff_bias_avg", "n_valid", "n_total"):
        row[k] = winner_metrics.get(k) if winner_metrics else None
    return row


def _append_history(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_history(csv_path, out_dir):
    """Overwrite fitness_per_round.png / metrics_per_round.png from history.csv (called after every round)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    genes = list(dict.fromkeys(df["gene"]))

    fig, axes = plt.subplots(len(genes), 1, figsize=(8, 2.6 * len(genes)), sharex=True, squeeze=False)
    for ax, g in zip(axes[:, 0], genes):
        d = df[df["gene"] == g]
        ax.plot(d["round"], d["fit_A"], marker="o", label="individual A")
        ax.plot(d["round"], d["fit_B"], marker="s", label="individual B")
        ax.plot(d["round"], d[["fit_A", "fit_B"]].max(axis=1), "k--", label="best")
        rep = d[d["replaced"] == True]  # noqa: E712
        ax.scatter(rep["round"], rep[["fit_A", "fit_B"]].max(axis=1), s=140, facecolors="none",
                   edgecolors="red", label="replaced")
        ax.set_ylabel("fitness")
        ax.set_title(g, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=4)
    axes[-1, 0].set_xlabel("round")
    axes[-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "fitness_per_round.png", dpi=130)
    plt.close(fig)

    per_round = df.dropna(subset=["acc_a"]).groupby("round")[["acc_a", "acc_d", "diff_bias_avg"]].last()
    if len(per_round):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for col in per_round.columns:
            ax.plot(per_round.index, per_round[col], marker="o", label=col)
        ax.set_xlabel("round")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("score (Evolve set)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "metrics_per_round.png", dpi=130)
        plt.close(fig)


def plot_select_summary(baseline, evolved, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = ["acc_a", "acc_d", "acc_avg", "diff_bias_a", "diff_bias_d", "diff_bias_avg"]
    x = range(len(keys))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar([i - 0.2 for i in x], [baseline[k] for k in keys], width=0.4, label="baseline")
    ax.bar([i + 0.2 for i in x], [evolved[k] for k in keys], width=0.4, label="evolved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=20, fontsize=8)
    ax.set_title("Select set (held out from evolution)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "select_set_summary.png", dpi=130)
    plt.close(fig)
