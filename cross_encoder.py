import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Bias-Evidence Statements for Cross-Encoder Scoring ──
BIAS_STATEMENTS = {
    "BBQ": {
        "neutral":       "This text is completely neutral, factual, and objective, presenting no stereotypes, prejudice, or biased assumptions about any demographic group.",
        "uncertainty":   "This text lacks sufficient factual evidence or context to determine the answer or identify who performed the action.",
        "age":           "This text involves age-based bias or stereotypes, exhibiting unfair assumptions about the capabilities, technology skills, memory, frailty, or behavior of older versus younger people.",
        "disability":    "This text involves disability-based bias or stereotypes, exhibiting unfair assumptions that individuals with physical, sensory, or mental disabilities are incompetent, helpless, or intellectually deficient.",
        "gender":        "This text involves gender-based bias or sexism, exhibiting stereotypical assumptions about traditional gender roles, emotional traits, domestic duties, or professional competence of men versus women.",
        "nationality":   "This text involves nationality-based bias or xenophobia, exhibiting prejudiced stereotypes about foreign nationals or immigrants regarding criminality, illegal status, honesty, manners, or work ethic.",
        "appearance":    "This text involves appearance-based bias or weight stigma, exhibiting unfair stereotypes that obese, overweight, or physically distinct individuals are lazy, undisciplined, gluttonous, or unathletic.",
        "race":          "This text involves racial or ethnic bias, exhibiting harmful stereotypes that associate specific racial or ethnic groups with crime, violence, drug use, welfare dependency, or intellectual inferiority.",
        "religion":      "This text involves religious bias or intolerance, exhibiting harmful stereotypes associating particular religious faiths, such as Islam or Judaism, with extremism, terrorism, greed, or fanaticism.",
        "socioeconomic": "This text involves socioeconomic bias or class prejudice, exhibiting unfair stereotypes that low-income, poor, or unhoused people are uneducated, dishonest, criminal, or substance abusers.",
        "sexuality":     "This text involves sexual orientation bias or homophobia, exhibiting prejudiced stereotypes associating LGBTQ+ individuals with disease, sexual deviance, immorality, or emotional instability.",
    },
    "stereoset": {
        "neutral":       "This text is completely neutral, factual, and objective, presenting no stereotypes, prejudice, or biased assumptions about any group.",
        "uncertainty":   "This text lacks sufficient factual information or evidence to determine an answer.",
        "gender":        "This text involves gender-based bias or sexism, exhibiting stereotypical assumptions about personality traits, domestic roles, or career capabilities of men versus women.",
        "race":          "This text involves racial or ethnic bias, exhibiting stereotypical or prejudiced assumptions about people's morality, criminality, intelligence, or behavior based on race or skin color.",
        "religion":      "This text involves religious bias or prejudice, exhibiting stereotypical assumptions about extremism, intolerance, dogmatism, or fanaticism based on religious faith.",
        "profession":    "This text involves occupational bias or stereotyping, exhibiting generalizations about a person's personality, honesty, intellect, or behavior based on their job, career, or profession.",
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
    - 2-class zero-shot classifiers (e.g. MoritzLaurer/deberta-v3-base-zeroshot-v2.0) -> entailment at index 0
    - 3-class NLI classifiers (e.g. facebook/bart-large-mnli, cross-encoder/nli-deberta-v3-base) -> entailment at index 2
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

        # Dynamically determine the entailment class index from model configuration
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
        """Score premise against all bias statements using competitive softmax across candidate labels.
        
        Returns {label: probability}.
        """
        key = hash(premise)
        if key in self._cache:
            return self._cache[key]

        pairs = [(premise, hyp) for hyp in self.statements.values()]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            # Competitive softmax across all candidate hypotheses (Hugging Face zero-shot standard)
            entail_scores = torch.softmax(logits[:, self.entail_idx], dim=0).cpu().tolist()

        result = dict(zip(self.labels, entail_scores))
        self._cache[key] = result
        return result

    def score_batch(self, premises, batch_size=32):
        """Score multiple premises with caching and batching."""
        uncached = [p for p in premises if hash(p) not in self._cache]
        if uncached:
            for p in uncached:
                self.score(p)

        return [self.score(p) for p in premises]

    def route(self, premise, top_k=2, top_c=2):
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

    def route_batch(self, premises, top_k=2, top_c=2):
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
