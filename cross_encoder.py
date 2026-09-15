import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Topic-Matching Statements for Cross-Encoder Scoring ──
BIAS_STATEMENTS = {
    "BBQ": {
        "neutral":       "This text does not involve any sensitive social group or demographic.",
        "uncertainty":   "This text does not provide enough information to determine a definitive answer.",
        "age":           "This text involves people of different ages or generations.",
        "disability":    "This text involves people with different ability or disability statuses.",
        "gender":        "This text involves people of different genders or gender identities.",
        "nationality":   "This text involves people of different nationalities or ethnic origins.",
        "appearance":    "This text involves people with different physical appearances.",
        "race":          "This text involves people of different races or ethnicities.",
        "religion":      "This text involves people of different religions or beliefs.",
        "socioeconomic": "This text involves people of different socioeconomic backgrounds.",
        "sexuality":     "This text involves people of different sexual orientations.",
    },
    "stereoset": {
        "neutral":       "This text does not involve any sensitive social group or demographic.",
        "uncertainty":   "This text does not provide enough information to determine a definitive answer.",
        "gender":        "This text involves people of different genders or gender identities.",
        "race":          "This text involves people of different races or ethnicities.",
        "religion":      "This text involves people of different religions or beliefs.",
        "profession":    "This text involves people of different professions or occupations.",
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
    """Routes samples to Fast Path or Adaptive Persona Path using NLI entailment scores."""

    def __init__(self, model_name, device, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.statements = BIAS_STATEMENTS[dataset]
        self.labels = list(self.statements.keys())
        self._cache = {}

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
            # NLI output: [contradiction, neutral, entailment] → use entailment (idx 2)
            entail_scores = torch.softmax(logits, dim=-1)[:, 2].cpu().tolist()

        result = dict(zip(self.labels, entail_scores))
        self._cache[key] = result
        return result

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
        return [self.route(p, top_k, top_c) for p in premises]


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
