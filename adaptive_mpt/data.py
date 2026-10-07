import json
import random
import re
from pathlib import Path
from xopen import xopen

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

def set_tasks_BiasFreeBench():
    return [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Religion",
        "SES",
        "Sexual_orientation",
    ]

UNDETERMINED_PHRASES = {
    "unknown", "undetermined", "not known", "not answerable",
    "cannot be determined", "can't be determined",
    "cannot answer", "can't answer",
    "not enough info", "not enough information",
}

def _normalize_option(text):
    return str(text).strip().rstrip(".").strip().lower()

# StereoSet targets are not always people (countries, scriptures, ...), so pick a placeholder per category
STEREOSET_PLACEHOLDERS = {
    "gender": "[Person A]",
    "race": "[Group A]",
    "religion": "[Religion A]",
    "profession": "[Profession A]",
}

GENDERED_WORDS = {
    "man": "person", "men": "people", "woman": "person", "women": "people",
    "boy": "child", "boys": "children", "girl": "child", "girls": "children",
    "gentleman": "person", "gentlemen": "people", "lady": "person", "ladies": "people",
    "guy": "person", "guys": "people",
    "mom": "parent", "moms": "parents", "mum": "parent", "mother": "parent", "mothers": "parents",
    "dad": "parent", "dads": "parents", "father": "parent", "fathers": "parents",
    "mommy": "parent", "daddy": "parent",
    "grandmother": "grandparent", "grandfather": "grandparent", "grandma": "grandparent", "grandpa": "grandparent",
    "husband": "spouse", "wife": "spouse", "husbands": "spouses", "wives": "spouses",
    "boyfriend": "partner", "girlfriend": "partner",
    "son": "child", "daughter": "child", "sons": "children", "daughters": "children",
    "brother": "sibling", "sister": "sibling", "brothers": "siblings", "sisters": "siblings",
    "uncle": "relative", "aunt": "relative", "nephew": "relative", "niece": "relative",
    "himself": "themself", "herself": "themself",
}

# Words after "her" that mean it is an object pronoun ("told her that"), not a possessive ("her car")
_HER_OBJECT_FOLLOWERS = {
    "and", "or", "but", "to", "with", "for", "at", "in", "on", "a", "an", "the", "that", "as", "so",
    "because", "up", "out", "off", "down", "back", "again", "too", "very", "more", "about", "from",
    "by", "into", "over", "when", "if", "is", "was", "be", "how", "what", "why", "it", "this",
}

def _match_case(word, repl):
    return repl[0].upper() + repl[1:] if word[:1].isupper() else repl

def neutralize_gender(text, ph):
    """Replace gendered pronouns with the placeholder and gendered nouns with neutral ones."""
    text = re.sub(r"\b(?:he|she|him)\b", ph, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:his|hers)\b", f"{ph}'s", text, flags=re.IGNORECASE)

    def her(m):
        nxt = m.group(1)
        if nxt and nxt.lower() not in _HER_OBJECT_FOLLOWERS:
            return f"{ph}'s {nxt}"
        return ph + (f" {nxt}" if nxt else "")
    text = re.sub(r"\bher\b(?:\s+([A-Za-z]+))?", her, text, flags=re.IGNORECASE)

    pattern = re.compile(r"\b(" + "|".join(map(re.escape, GENDERED_WORDS)) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda m: _match_case(m.group(1), GENDERED_WORDS[m.group(1).lower()]), text)

def _contains(text, phrase):
    return re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE) is not None

def _split_name(name, context):
    """Split an answer_info name that does not appear verbatim in the context into (name, descriptor).

    BBQ Race_x_SES stores e.g. "Juan Rodriguez physician" while the context says
    "Juan Rodriguez, who's a physician"; take the longest word prefix that occurs in the context.
    """
    if _contains(context, name):
        return name, None
    words = name.split()
    for k in range(len(words) - 1, 0, -1):
        head, tail = " ".join(words[:k]), " ".join(words[k:])
        if _contains(context, head):
            return head, (tail if _contains(context, tail) else None)
    return name, None

def _substitute(text, name_map, suffix):
    """Replace all names in one pass, so a name like "person" never matches inside an inserted "[Person A]"."""
    if not name_map:
        return text
    lookup = {name.lower(): ph for name, ph in name_map.items()}
    alternatives = "|".join(re.escape(name) for name in sorted(name_map, key=len, reverse=True))
    pattern = re.compile(rf"\b({alternatives})({suffix})?\b", re.IGNORECASE)

    def repl(m):
        base, suf = m.group(1), (m.group(2) or "").lower()
        # "The cook cooks": a plural form right after the same noun is a verb, keep it
        if suf in ("s", "es") and re.search(rf"\b{re.escape(base)}\s+$", m.string[:m.start()], re.IGNORECASE):
            return m.group(0)
        # Keep the possessive so "Juan's car" becomes "[Person A]'s car"
        return lookup[base.lower()] + ("'s" if suf == "'s" else "")

    return pattern.sub(repl, text)

def mask_entities(item):
    """Anonymize character identities for objective logic checks.

    - BBQ: Masks the people in the answer options into [Person A], [Person B], and SES descriptors
      that are stored together with names (Race_x_SES) into [Role A], [Role B].
    - StereoSet: Masks the target with a category-specific placeholder; for gender also neutralizes
      gendered pronouns and nouns in the context and options.
    """
    masked_item = dict(item)
    ans_info = item.get("answer_info", {})
    name_map = {}
    suffix = r"'s|s|es"
    neutralize = None

    if ans_info:
        context_text = str(item.get("context", ""))
        names, roles = [], []
        for ans_key in ("ans0", "ans1", "ans2"):
            if ans_key in ans_info:
                val = ans_info[ans_key]
                name = val[0] if isinstance(val, (list, tuple)) else str(val)
                label = val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else ""
                # Use the annotation label, not the surface text: BBQ has 10 phrasings of the unknown option
                if label == "unknown" or _normalize_option(name) in UNDETERMINED_PHRASES:
                    continue
                # Some BBQ Race_x_gender rows have empty names; recover them from the option text
                if not name.strip():
                    name = re.sub(r"^the\s+", "", str(item.get(ans_key, "")).strip(), flags=re.IGNORECASE)
                if not name.strip():
                    continue
                person, role = _split_name(name.strip(), context_text)
                if person not in names:
                    names.append(person)
                if role and role not in roles:
                    roles.append(role)
        for idx, name in enumerate(names):
            name_map[name] = f"[Person {chr(65 + idx)}]"
        for idx, role in enumerate(roles):
            name_map.setdefault(role, f"[Role {chr(65 + idx)}]")
        # Longest first so "Juan Rodriguez" is replaced before a shorter overlapping string
        name_map = dict(sorted(name_map.items(), key=lambda kv: -len(kv[0])))

    elif "target" in item and item["target"]:
        target = str(item["target"]).strip()
        category = str(item.get("category", "")).lower()
        ph = STEREOSET_PLACEHOLDERS.get(category, "[Person A]")
        name_map[target] = ph
        # Race targets are often countries used as demonyms ("Bolivia" -> "Bolivian", "Austrians")
        suffix = r"'s|s|es|n|an|ns|ans" if category == "race" else r"'s|s|es"
        if category == "gender":
            neutralize = lambda text: neutralize_gender(text, ph)

    def mask_text(text):
        text = _substitute(str(text), name_map, suffix)
        return neutralize(text) if neutralize else text

    context = mask_text(masked_item.get("context", ""))
    question = mask_text(masked_item.get("question", ""))

    masked_item["context"] = context
    masked_item["question"] = question
    masked_item["_entity_map"] = name_map

    if ans_info:
        new_ans_info = {}
        for ans_key, val in ans_info.items():
            if isinstance(val, (list, tuple)):
                old_name = val[0]
                new_name = name_map.get(old_name, old_name)
                new_ans_info[ans_key] = [new_name] + list(val[1:])
            else:
                new_ans_info[ans_key] = name_map.get(str(val), val)
        masked_item["answer_info"] = new_ans_info

    for opt_key in ("ans0", "ans1", "ans2"):
        if opt_key in masked_item:
            masked_item[opt_key] = mask_text(masked_item[opt_key])

    return masked_item

def get_undetermined_index(item):
    """Identify the choice index corresponding to 'Cannot be determined'."""
    ans_info = item.get("answer_info", {})
    for key, val in ans_info.items():
        label = val[1] if isinstance(val, (list, tuple)) and len(val) > 1 else str(val)
        if label == "unknown":
            match = re.search(r'\d+', key)
            if match:
                return int(match.group())
    # Prefer an option that is exactly an unknown phrase, so sentences merely containing "unknown" are not picked
    for opt_key in ("ans0", "ans1", "ans2"):
        if opt_key in item and _normalize_option(item[opt_key]) in UNDETERMINED_PHRASES:
            return int(opt_key[-1])
    undetermined_keywords = [
        "cannot be determined", "can't be determined", "not enough info",
        "undetermined", "not answerable", "cannot answer", "can't answer",
        "not known", "unknown"
    ]
    for opt_key in ("ans0", "ans1", "ans2"):
        if opt_key in item:
            text = str(item[opt_key]).lower()
            if any(w in text for w in undetermined_keywords):
                match = re.search(r'\d+', opt_key)
                if match:
                    return int(match.group())
    return -1

def is_classified_as_ambiguous(item, pred):
    """Check whether a prediction index maps to the undetermined option."""
    return pred == get_undetermined_index(item)

def load_dataset_records(args, batch_input_paths):
    """Load dataset records and apply deterministic sampling."""
    dataset = args.dataset
    models_name = "_".join(dict.fromkeys(args.models))
    sample_records = []
    category_manifest = []

    data_split = getattr(args, "data_split", None)
    split_ratio = None
    split_part = None
    if data_split is not None:
        split_ratio = float(data_split[0])
        split_part = int(data_split[1])
        if not (0.0 < split_ratio <= 1.0):
            raise ValueError(f"Split ratio must be in (0, 1], got {split_ratio}")
        if split_part < 1:
            raise ValueError(f"Split part index must be >= 1, got {split_part}")

    for input_path in batch_input_paths:
        category = Path(input_path).stem
        output_dir = Path("predictions") / dataset / models_name / args.note / args.method
        output_dir.mkdir(parents=True, exist_ok=True)
        
        split_suffix = f"_part{split_part}" if split_part is not None else ""
        output_file = output_dir / f"preds_{category}{split_suffix}.jsonl"

        if output_file.exists() and not getattr(args, "overwrite", False) and split_part is None:
            records = [json.loads(line) for line in xopen(output_file, "r", encoding="utf-8")]
        else:
            records = [json.loads(line) for line in xopen(input_path, "r", encoding="utf-8")]

        if args.truncate:
            if dataset == "BBQ":
                len_groups = 20
                original_count = len(records)
                example_groups = []
                for i in range(0, len(records), 4):
                    if i + 3 < len(records):
                        example_groups.append(records[i:i+4])
                print(f"Task: {category}, Total groups: {len(example_groups)}, Original samples: {original_count}")
                if len(example_groups) > len_groups:
                    selected_groups = random.sample(example_groups, len_groups)
                    records = []
                    for group in selected_groups:
                        records.extend(group)
                    print(f"Task: {category}, Selected groups: {len_groups}, Sampled samples: {len(records)}")
                else:
                    records = []
                    for group in example_groups:
                        records.extend(group)
                    print(f"Task: {category}, Using all groups: {len(example_groups)}, Total samples: {len(records)}")

            elif dataset == "stereoset":
                len_samples = 100
                records = random.sample(records, len_samples)
                print(f"Task: {category}, Selected examples: {len_samples}, Total samples: {len(records)}")

            elif dataset in ("BiasFreeBench", "biasfreebench"):
                len_samples = 100
                records = random.sample(records, min(len_samples, len(records)))
                print(f"Task: {category}, Selected examples: {len(records)}, Total samples: {len(records)}")

        # Apply data_split slicing
        if split_ratio is not None and split_part is not None:
            total_n = len(records)
            start_idx = int((split_part - 1) * split_ratio * total_n)
            if split_part * split_ratio < 1.0:
                end_idx = min(int(split_part * split_ratio * total_n), total_n)
            else:
                end_idx = total_n

            if dataset == "BBQ":
                start_idx = (start_idx // 4) * 4
                if split_part * split_ratio < 1.0:
                    end_idx = (end_idx // 4) * 4
                else:
                    end_idx = total_n

            if start_idx >= total_n:
                raise ValueError(f"Split part {split_part} is out of bounds for ratio {split_ratio} (total available: {total_n})")

            orig_len = len(records)
            records = records[start_idx:end_idx]
            print(f"Task: {category}, Split [{split_ratio}, Part {split_part}]: Sliced [{start_idx}:{end_idx}] ({len(records)}/{orig_len} samples)")

        for line in records:
            line['_task_category'] = category
            line['_output_file'] = str(output_file)

        sample_records.extend(records)
        category_manifest.append((category, len(records), output_file))

    return sample_records, category_manifest

def save_dataset_records(sample_records, category_manifest):
    """Write evaluation results back to per-task JSONL files."""
    current_idx = 0
    for category, count, output_file in category_manifest:
        task_records = sample_records[current_idx:current_idx + count]
        for line in task_records:
            line.pop('_task_category', None)
            line.pop('_output_file', None)
            line.pop('_ce_scores', None)
            line.pop('_ce_labels', None)

        with xopen(output_file, "w", encoding="utf-8") as fout:
            for line in task_records:
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"Results for {category} saved to {output_file}")
        current_idx += count
