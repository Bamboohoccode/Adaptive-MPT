import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Topic-Matching Statements for Cross-Encoder Scoring ──
# Affirmative formulation to avoid negation blindness in NLI models
BIAS_STATEMENTS = {
    "BBQ": {
        "neutral":       "This text is an objective, neutral statement without demographic stereotypes or social bias.",
        "uncertainty":   "There is not enough information in the text to determine a definitive answer.",
        "age":           "This text involves age groups, elderly people, or age stereotypes.",
        "disability":    "This text involves disability status, mental health, or physical challenges.",
        "gender":        "This text involves gender identity, men, women, or gender stereotypes.",
        "nationality":   "This text involves nationalities, foreign countries, immigrants, or ethnic origins.",
        "appearance":    "This text involves physical appearance, body weight, looks, or beauty standards.",
        "race":          "This text involves race, African American, Latino, Asian, or racial stereotypes.",
        "religion":      "This text involves religion, religious beliefs, Islam, Judaism, or Christianity.",
        "socioeconomic": "This text involves socioeconomic status, wealth, poverty, low income, or social class.",
        "sexuality":     "This text involves sexual orientation, gay, lesbian, bisexual, or LGBTQ.",
    },
    "stereoset": {
        "neutral":       "This text is an objective, neutral statement without demographic stereotypes or social bias.",
        "uncertainty":   "There is not enough information in the text to determine a definitive answer.",
        "gender":        "This text involves gender identity, men, women, or gender stereotypes.",
        "race":          "This text involves race, ethnicity, nationality, or racial stereotypes.",
        "religion":      "This text involves religion, religious beliefs, or religious stereotypes.",
        "profession":    "This text involves professions, careers, jobs, or occupational stereotypes.",
    },
}

# ── Label → Persona Description Mapping ──
LABEL_TO_PERSONA = {
    "age":           "person from a different age group",
    "disability":    "person with a different ability status",
    "gender":        "person of a different gender identity",
    "nationality":   "person of a different nationality",
    "appearance":    "person with a different physical appearance",
    "race":          "person of a different racial or ethnic background",
    "religion":      "person of a different religious belief",
    "socioeconomic": "person from a different socioeconomic background",
    "sexuality":     "person of a different sexual orientation",
    "profession":    "person from a different profession",
}


class CrossEncoderRouter:
    """Routes samples to Fast Path or Adaptive Persona Path using NLI entailment scores.
    
    Supports:
    - 2-class zero-shot classifiers (e.g. MoritzLaurer/deberta-v3-base-zeroshot-v2.0)
    - 3-class NLI classifiers (e.g. facebook/bart-large-mnli, cross-encoder/nli-deberta-v3-base)
    """

    def __init__(self, model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", device="cpu", dataset="BBQ"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.statements = BIAS_STATEMENTS[dataset]
        self.labels = list(self.statements.keys())
        self._cache = {}

        # Dynamically determine the entailment class index based on model config
        self.entail_idx = self._find_entailment_index()

    def _find_entailment_index(self):
        """Detect index of entailment class from model config."""
        if hasattr(self.model.config, "label2id") and self.model.config.label2id:
            for name, idx in self.model.config.label2id.items():
                if "entail" in name.lower() and "not" not in name.lower():
                    return idx
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            for idx, name in self.model.config.id2label.items():
                if "entail" in name.lower() and "not" not in name.lower():
                    return int(idx)
        # Fallback: 2-class zero-shot models (like MoritzLaurer) use 0, 3-class MNLI models use 2
        num_labels = getattr(self.model.config, "num_labels", 2)
        return 0 if num_labels == 2 else 2

    def score(self, premise):
        """Score premise against all bias statements. Returns {label: entailment_score}."""
        key = hash(premise)
        if key in self._cache:
            return self._cache[key]

        pairs = [(premise, hyp) for hyp in self.statements.values()]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            entail_scores = torch.softmax(logits, dim=-1)[:, self.entail_idx].cpu().tolist()

        result = dict(zip(self.labels, entail_scores))
        self._cache[key] = result
        return result

    def score_batch(self, premises, batch_size=32):
        """Score multiple premises with caching and batching."""
        uncached = [p for p in premises if hash(p) not in self._cache]
        if uncached:
            for i in range(0, len(uncached), batch_size):
                chunk = uncached[i : i + batch_size]
                all_pairs = []
                for p in chunk:
                    for hyp in self.statements.values():
                        all_pairs.append((p, hyp))
                inputs = self.tokenizer(
                    all_pairs, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    entail_scores = torch.softmax(logits, dim=-1)[:, self.entail_idx].cpu().tolist()

                num_labels = len(self.labels)
                for j, p in enumerate(chunk):
                    start = j * num_labels
                    end = start + num_labels
                    self._cache[hash(p)] = dict(zip(self.labels, entail_scores[start:end]))

        return [self.score(p) for p in premises]

    def route(self, premise, top_k, top_c):
        """Route a single premise.

        Returns:
            is_fast: True if "neutral" is within the top-c ranked labels.
            bias_labels: Top-k non-neutral/non-uncertainty labels for persona selection.
            scores: Full {label: score} dict for logging.
        """
        scores = self.score(premise)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_c_labels = [label for label, _ in ranked[:top_c]]
        is_fast = "neutral" in top_c_labels
        bias_labels = [l for l, _ in ranked if l not in ("neutral", "uncertainty")][:top_k]
        return is_fast, bias_labels, scores

    def route_batch(self, premises, top_k, top_c):
        """Route a batch of premises."""
        all_scores = self.score_batch(premises)
        results = []
        for scores in all_scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_c_labels = [label for label, _ in ranked[:top_c]]
            is_fast = "neutral" in top_c_labels
            bias_labels = [l for l, _ in ranked if l not in ("neutral", "uncertainty")][:top_k]
            results.append((is_fast, bias_labels, scores))
        return results


def save_ce_cache(router, path):
    """Save Cross-Encoder score cache to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for k, v in router._cache.items():
            f.write(json.dumps({"key": k, "scores": v}) + "\n")


def load_ce_cache(router, path):
    """Load Cross-Encoder score cache from JSONL."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            router._cache[d["key"]] = d["scores"]
