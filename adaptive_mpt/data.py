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

def mask_entities(item):
    """Anonymize character identities to [Person A], [Person B] for objective logic checks.
    
    - BBQ: Masks demographic pairs into [Person A] and [Person B].
    - StereoSet: Masks target identity/profession into [Person A].
    """
    masked_item = dict(item)
    ans_info = item.get("answer_info", {})
    name_map = {}

    if ans_info:
        names = []
        for ans_key in ("ans0", "ans1", "ans2"):
            if ans_key in ans_info:
                val = ans_info[ans_key]
                name = val[0] if isinstance(val, (list, tuple)) else str(val)
                if name.lower() not in ("unknown", "cannot be determined", "can't be determined", "not answerable", "undetermined"):
                    names.append(name)
        names = sorted(set(names), key=lambda x: -len(x))
        placeholders = ["[Person A]", "[Person B]", "[Person C]"]
        for idx, name in enumerate(names):
            name_map[name] = placeholders[idx] if idx < len(placeholders) else f"[Person {chr(65+idx)}]"

    elif "target" in item and item["target"]:
        target = str(item["target"]).strip()
        name_map[target] = "[Person A]"

    context = str(masked_item.get("context", ""))
    question = str(masked_item.get("question", ""))
    for name, ph in name_map.items():
        pattern = re.compile(rf"\b{re.escape(name)}(?:'s|s|es)?\b", re.IGNORECASE)
        context = pattern.sub(ph, context)
        question = pattern.sub(ph, question)

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
            text = str(masked_item[opt_key])
            for name, ph in name_map.items():
                pattern = re.compile(rf"\b{re.escape(name)}(?:'s|s|es)?\b", re.IGNORECASE)
                text = pattern.sub(ph, text)
            masked_item[opt_key] = text

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
