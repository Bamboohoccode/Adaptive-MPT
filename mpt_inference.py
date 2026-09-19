import argparse
import numpy as np
import random
import torch
import json
import os
import time
from openai import OpenAI
import re
from tqdm import tqdm
from xopen import xopen
from collections import Counter, defaultdict
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts import (
    BEG_LLAMA3, SYS_LLAMA3, USR_LLAMA3, AST_LLAMA3, END_LLAMA3,
    SYS_QWEN, USR_QWEN, AST_QWEN, END_QWEN,
    DEFAULT_SYS_PROMPT, PERSONA_SYS_PROMPT,
    QUESTION_PROMPT_BBQ, QUESTION_PROMPT_STEREOSET,
    DEBIAS_PROMPT, PERSONA_PROMPT, REVIEW_PROMPT, SUMMARY_PROMPT, POSTFIX_PROMPT,
    CONSTRAINT_PROMPT_BBQ, CONSTRAINT_PROMPT_STS,
    UNCERTAINTY_WARNING, BIAS_WARNING, LOGICAL_PROMPT, REPROMPT,
    LOGICAL_PROMPT_MASKED, REPROMPT_MASKED, REVIEW_PROMPT_MASKED, SUMMARY_PROMPT_MASKED,
    LOGICAL_PROMPT_FACT_EXTRACTION, REVIEW_PROMPT_FACT_EXTRACTION,
    SUMMARY_PROMPT_FACT_EXTRACTION, REPROMPT_FACT_EXTRACTION,
    LOGICAL_PROMPT_ASYMMETRIC_UNMASKED,
    FACT_ANCHOR_SYS_PROMPT, BIAS_AUDITOR_SYS_PROMPT, JUDGE_PERSONA_SYS_PROMPT,
)
from cross_encoder import (
    CrossEncoderRouter, LABEL_TO_PERSONA,
    load_ce_cache, save_ce_cache,
)


class Decoder():
    def __init__(self, args, model_name):
        self.model_name = model_name

        self.inference_type = args.inference_type
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_tokens = args.max_tokens
        self.is_chat_templates = args.chat_templates
        self.args = args
        self.call_count = 0
        self.total_prompts_decoded = 0

        if self.is_chat_templates:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                # cache_dir=args.models_dir
                )

        if self.inference_type == "vllm":
            num_gpus = getattr(args, "num_gpus", 1)
            gpu_mem = getattr(args, "gpu_memory_utilization", 0.85)
            max_len = getattr(args, "max_model_len", 32768)
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=gpu_mem,
                # download_dir=args.models_dir,
                max_model_len=max_len,
            )
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
        
        elif self.inference_type == "openai":
            self.model = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
    
    def decode(self, prompt):
        self.call_count += 1
        num_prompts = len(prompt) if isinstance(prompt, list) else 1
        self.total_prompts_decoded += num_prompts

        if self.inference_type == "vllm":
            if self.is_chat_templates:
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

            if hasattr(self.args, "log_path") and self.args.log_path:
                log_data(f"{prompt[0]}", self.args)

            outputs = self.model.generate(
                prompt,
                self.sampling_params,
                use_tqdm=True
            )
            return outputs

        elif self.inference_type == "openai":
            try:
                outputs = self.model.chat.completions.create(
                    model=self.model_name,    
                    messages=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    stop=None,
                    n=1,       
                )

                assert outputs is not None, "API returned None"
                assert hasattr(outputs, 'choices'), f"Response missing 'choices' attribute: {outputs}"
                assert len(outputs.choices) > 0, f"No choices in response: {outputs}"
                assert hasattr(outputs.choices[0], 'message'), f"Choice missing 'message' attribute: {outputs.choices[0]}"
                assert hasattr(outputs.choices[0].message, 'content'), f"Message missing 'content' attribute: {outputs.choices[0].message}"
                
                return outputs

            except Exception as e:
                print(f"\nRetrying due to an error: {e}\n")
                time.sleep(5)

                return self.decode(prompt)
            

def set_tasks_BBQ():
    tasks = [
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
    return tasks

def set_tasks_stereoset():
    tasks = [
        "gender_inter",
        "gender_intra",
        "race_inter",
        "race_intra",
        "religion_inter",
        "religion_intra",
        "profession_inter",
        "profession_intra",
    ]
    return tasks

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def log_start(METHOD, args):
    models_str = "_".join(dict.fromkeys(args.models))
    log_path = f"logs/{args.dataset}/{models_str}/{METHOD}"
    try:
        os.makedirs(log_path)
    except:
        log_path += f"_{time.strftime('%m_%d_%H_%M', time.localtime())}"
        os.makedirs(log_path)

    with open(log_path + '/python_file.py', 'a', encoding='utf-8') as f:
        with open(os.path.basename(__file__), 'r', encoding='utf-8') as f2:
            file = f2.read()
        f.write(file)

    args.log_path = log_path

def log_data(text, args):
    log_path = getattr(args, "log_path", None)
    if not log_path:
        return
    try:
        os.makedirs(log_path, exist_ok=True)
        with open(log_path + '/loggings.txt', 'a', encoding='utf-8') as f:
            f.write(str(text))
            f.write('\n')
    except Exception:
        pass

def get_groups(args, line):
    """Legacy heuristic group extraction. Only used when --disable_cross_encoder is set."""
    dataset = args.dataset
    if dataset == "BBQ":
        if line["category"] == "Nationality":
            groups = [grp for (grp, group) in line["answer_info"].values() if group.lower() != "unknown"]
        else:
            groups = [group for (_, group) in line["answer_info"].values() if group.lower() != "unknown"]
    
    elif dataset == "stereoset":
        target = line["target"]
        if line["category"] == "gender":
            groups = ["self-identified male", "self-identified female"]
        else:
            groups = [f"{target}", f"non-{target}"]

    return groups

def get_personas_from_labels(bias_labels):
    """Map Cross-Encoder detected bias labels to persona description strings."""
    return [LABEL_TO_PERSONA[label] for label in bias_labels if label in LABEL_TO_PERSONA]

def mask_entities(item):
    """Mask character entities in context, question, and options with [Person A] and [Person B]
    to decouple demographic stereotypes from logical reasoning.
    """
    # Skip StereoSet where options are complete sentences rather than individual entities
    if item.get("category") in ("gender_inter", "gender_intra", "race_inter", "race_intra",
                                "religion_inter", "religion_intra", "profession_inter", "profession_intra"):
        return item

    masked_item = dict(item)
    ans_info = masked_item.get("answer_info", {})
    non_unknown = []
    
    for k in ["ans0", "ans1", "ans2"]:
        if k in ans_info and isinstance(ans_info[k], (list, tuple)) and len(ans_info[k]) > 1:
            if str(ans_info[k][1]).lower() != "unknown":
                non_unknown.append(k)
        elif k in masked_item:
            val_lower = str(masked_item[k]).lower()
            if not any(kw in val_lower for kw in [
                "cannot be determined", "can't be determined",
                "not enough", "undetermined", "not answerable",
                "unknown", "not determinable", "cannot answer", "can't answer"
            ]):
                if k not in non_unknown:
                    non_unknown.append(k)
    
    if len(non_unknown) == 2:
        k_a, k_b = non_unknown[0], non_unknown[1]
        term_a = ans_info[k_a][0] if k_a in ans_info and isinstance(ans_info[k_a], (list, tuple)) and ans_info[k_a] else masked_item[k_a]
        term_b = ans_info[k_b][0] if k_b in ans_info and isinstance(ans_info[k_b], (list, tuple)) and ans_info[k_b] else masked_item[k_b]
        
        ctx = str(masked_item.get("context", ""))
        q = str(masked_item.get("question", ""))

        # Collect all replacement targets and sort descending by length to prevent substring collision
        # (e.g. "African American" must be replaced before "African")
        targets = []
        for term, raw_val, label in [
            (term_a, masked_item.get(k_a, ""), "[Person A]"),
            (term_b, masked_item.get(k_b, ""), "[Person B]"),
        ]:
            for t in [raw_val, term]:
                if t and isinstance(t, str) and len(t.strip()) > 1:
                    clean_t = re.sub(r"^(the|a|an)\s+", "", t.strip(), flags=re.IGNORECASE)
                    for target in [t.strip(), clean_t]:
                        if target and len(target) > 1:
                            targets.append((target, label))

        # Sort longest first to avoid substring prefix masking
        targets.sort(key=lambda x: len(x[0]), reverse=True)

        for target_str, label in targets:
            pattern = re.compile(rf"\b{re.escape(target_str)}\b", re.IGNORECASE)
            ctx = pattern.sub(label, ctx)
            q = pattern.sub(label, q)
        
        masked_item["context"] = ctx
        masked_item["question"] = q
        # Strictly mask options to prevent demographic identity leakage into Pass 1 prompts
        masked_item[k_a] = "[Person A]"
        masked_item[k_b] = "[Person B]"
        
    return masked_item

def get_undetermined_index(item):
    """Extract index (0, 1, or 2) corresponding to undetermined option across BBQ & StereoSet."""
    ainfo = item.get("answer_info", {})
    for k, v in ainfo.items():
        if len(v) > 1 and str(v[1]).lower() == "unknown":
            return int(k.replace("ans", ""))

    undetermined_keywords = [
        "cannot be determined", "can't be determined",
        "undetermined", "unknown", "not enough information",
        "not enough info", "not known", "not answerable",
        "not determinable", "cannot answer", "can't answer"
    ]
    for idx in [0, 1, 2]:
        text = str(item.get(f"ans{idx}", "")).lower()
        if any(kw in text for kw in undetermined_keywords):
            return idx

    return -1

def is_classified_as_ambiguous(pred, item):
    """Check whether the prediction indicates insufficient context (Ambiguous -> Fast Path).
    Correctly handles permuted BBQ option positions (where undetermined can be 0, 1, or 2).
    """
    if pred is None or pred == -1:
        return True

    u_idx = get_undetermined_index(item)
    if pred == u_idx:
        return True

    ans_text = str(item.get(f"ans{pred}", "")).lower()
    undetermined_keywords = [
        "cannot be determined", "can't be determined",
        "undetermined", "not enough information",
        "unknown", "not determinable", "not answerable"
    ]
    return any(kw in ans_text for kw in undetermined_keywords)

def get_prompt(args, input_example, method, is_init=True, warnings="", logical=False, is_summary=False, masked=False, is_reprompt=False):
    dataset = args.dataset

    context = input_example["context"]
    question = input_example["question"]
    options = f"Options: (0): {input_example['ans0']}; (1): {input_example['ans1']}; (2): {input_example['ans2']}"

    if dataset == "BBQ":
        ask_prompt = CONSTRAINT_PROMPT_BBQ.format(question_prompt=QUESTION_PROMPT_BBQ, context=context, question=question, options=options)
    elif dataset == "stereoset":
        ask_prompt = CONSTRAINT_PROMPT_STS.format(question_prompt=QUESTION_PROMPT_STEREOSET, context=context, question=question, options=options)

    fact_ext = getattr(args, "feature_extraction", False)

    debate_mode = getattr(args, "debate_mode", "symmetric")

    if is_reprompt:
        prompt = REPROMPT_MASKED if masked else (REPROMPT_FACT_EXTRACTION if fact_ext else REPROMPT)
    elif is_init:
        # When entity-masked, 100% decouple from bias: omit all bias/uncertainty warnings
        prompt = ask_prompt if masked else (ask_prompt + warnings)
        if logical or masked:
            if masked:
                prompt += LOGICAL_PROMPT_MASKED
            elif debate_mode == "asymmetric_tribunal":
                prompt += LOGICAL_PROMPT_ASYMMETRIC_UNMASKED
            elif fact_ext:
                prompt += LOGICAL_PROMPT_FACT_EXTRACTION
            else:
                prompt += LOGICAL_PROMPT
    elif is_summary:
        prompt = SUMMARY_PROMPT_MASKED if masked else (SUMMARY_PROMPT_FACT_EXTRACTION if fact_ext else SUMMARY_PROMPT)
    else:
        prompt = REVIEW_PROMPT_MASKED if masked else (REVIEW_PROMPT_FACT_EXTRACTION if fact_ext else REVIEW_PROMPT)

    if args.inference_type == "openai":
        prompt += "Restrict your response to two sentences. "

    prompt += POSTFIX_PROMPT

    return prompt

def format_prompt(args, usr_prompt, sys_prompt=None, thinking_history=None, id=0):
    inference_type = args.inference_type
    is_chat_templates = args.chat_templates
    model_name = args.models[id] if inference_type == "openai" else args.models[0]

    if inference_type == "vllm" and not is_chat_templates:
        if model_name.startswith("meta-llama"):
            formatted_prompt = [BEG_LLAMA3]
            # formatted_prompt = []
        elif model_name.startswith("Qwen"):
            formatted_prompt = []
    else:
        formatted_prompt = []
    
    # add system prompt
    if inference_type == "vllm" and not is_chat_templates:
        if model_name.startswith("meta-llama"):
            formatted_prompt.extend([
                SYS_LLAMA3,
                sys_prompt,
                END_LLAMA3,
            ])
        elif model_name.startswith("Qwen"):
            formatted_prompt.extend([
                SYS_QWEN,
                sys_prompt,
                END_QWEN,
            ])
    else:
        formatted_prompt.extend([
            {"role": "system", "content": sys_prompt},
        ])

    if thinking_history:
        for i, round_history in enumerate(thinking_history):
            if len(round_history) > 1:
                init_prompt = round_history[0]
                if inference_type == "vllm" and not is_chat_templates:
                    if model_name.startswith("meta-llama"):
                        formatted_prompt.extend([
                            USR_LLAMA3,
                            init_prompt,
                            END_LLAMA3,
                            AST_LLAMA3,
                        ])
                    elif model_name.startswith("Qwen"):
                        formatted_prompt.extend([
                            USR_QWEN,
                            init_prompt,
                            END_QWEN,
                            AST_QWEN,
                        ])
                else:
                    formatted_prompt.extend([
                        {"role": "user", "content": init_prompt},
                    ])
                history = []
                for i, resp in enumerate(round_history[1:]):
                    history.append(f"Response {i}: \n{{{" ".join(resp.splitlines())}}}")
                if inference_type == "vllm" and not is_chat_templates:
                    if model_name.startswith("meta-llama"):
                            formatted_prompt.extend([
                            "\n".join(history),
                            END_LLAMA3,
                        ])
                    elif model_name.startswith("Qwen"):
                        formatted_prompt.extend([
                            "\n".join(history),
                            END_QWEN,
                        ])
                else:
                    formatted_prompt.extend([
                        {"role": "assistant", "content": "\n".join(history)},
                    ])

    # add current prompt
    if inference_type == "vllm" and not is_chat_templates:
        if model_name.startswith("meta-llama"):
            formatted_prompt.extend([
                USR_LLAMA3,
                usr_prompt,
                END_LLAMA3,
                AST_LLAMA3,
            ])
        elif model_name.startswith("Qwen"):
            formatted_prompt.extend([
                USR_QWEN,
                usr_prompt,
                END_QWEN,
                AST_QWEN,
            ])
    else:
        formatted_prompt.extend([
            {"role": "user", "content": usr_prompt},
        ])
    
    if inference_type == "vllm" and not is_chat_templates:
        return "\n".join(formatted_prompt)
    return formatted_prompt

def get_pred(resp):
    try:
        # 1. Match all <answer> tags and take the LAST tag (supports multi-agent CoT and self-correction)
        tag_matches = re.findall(r'<answer>(.*?)</answer>', resp, re.IGNORECASE | re.DOTALL)
        if tag_matches:
            inner_text = tag_matches[-1].strip()
            # Match digit 0, 1, 2 inside answer tags (e.g. "1", "ans1", "(1)", "Option 1")
            m = re.search(r'([0-2])', inner_text)
            if m:
                return int(m.group(1))
            # Match letter option: (A)->0, (B)->1, (C)->2
            m_letter = re.search(r'\b([A-C])\b', inner_text, re.IGNORECASE)
            if m_letter:
                return ord(m_letter.group(1).upper()) - ord('A')

        # 2. Check explicit final answer declaration near end of text (e.g. "Final Answer: 1")
        end_snippet = resp[-200:]
        m_decl = re.search(r'(?:final answer|correct option|choice)[:\s*]+([0-2]|[A-C])\b', end_snippet, re.IGNORECASE)
        if m_decl:
            val = m_decl.group(1)
            return int(val) if val.isdigit() else (ord(val.upper()) - ord('A'))

        # 3. Fallback: match digit only in final tail of text to avoid prompt numbers
        m = re.search(r'\b([0-2])\b', end_snippet)
        return int(m.group(1)) if m else -1

    except Exception:
        return -1


TOTAL_LLM_CALLS = 0


def decode_batch(models, args, formatted_prompts, **kwargs):
    """Unified decode for both vllm and openai backends with call tracking."""
    global TOTAL_LLM_CALLS
    if not formatted_prompts:
        return []

    TOTAL_LLM_CALLS += len(formatted_prompts)

    if args.inference_type == "vllm":
        raw = models[args.summary_id].decode(formatted_prompts)
        return [o.outputs[0].text.strip() for o in raw]
    elif args.inference_type == "openai":
        return [
            models[args.summary_id].decode(p).choices[0].message.content.strip()
            for p in tqdm(formatted_prompts, desc="openai inference")
        ]

def load_batch_data(args, batch_input_paths):
    """Load and prepare batch data. Returns (all_results, task_info)."""
    dataset = args.dataset
    models_name = "_".join(dict.fromkeys(args.models))
    all_results = []
    task_info = []

    for input_path in batch_input_paths:
        category = Path(input_path).stem
        output_dir = Path("predictions") / dataset / models_name / args.note / args.method
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"preds_{category}.jsonl"
        
        if output_file.exists():
            results = [json.loads(line) for line in xopen(output_file, "r", encoding="utf-8")]
        else:
            results = [json.loads(line) for line in xopen(input_path, "r", encoding="utf-8")]
        
        if args.truncate:
            if dataset == "BBQ":
                len_groups=20
                original_count = len(results)
                example_groups = []
                for i in range(0, len(results), 4):
                    if i + 3 < len(results):
                        group = results[i:i+4]
                        example_groups.append(group)
                print(f"Task: {category}, Total groups: {len(example_groups)}, Original samples: {original_count}")
                if len(example_groups) > len_groups:
                    selected_groups = random.sample(example_groups, len_groups)
                    results = []
                    for group in selected_groups:
                        results.extend(group)
                    print(f"Task: {category}, Selected groups: {len_groups}, Sampled samples: {len(results)}")
                else:
                    results = []
                    for group in example_groups:
                        results.extend(group)
                    print(f"Task: {category}, Using all groups: {len(example_groups)}, Total samples: {len(results)}")
            
            if dataset == "stereoset":
                len_samples = 100
                results = random.sample(results, len_samples)
                print(f"Task: {category}, Selected examples: {len_samples}, Total samples: {len(results)}")
        
        for line in results:
            line['_task_category'] = category
            line['_output_file'] = str(output_file)
        
        all_results.extend(results)
        task_info.append((category, len(results), output_file))

    return all_results, task_info

def write_results(all_results, task_info):
    """Write results back to per-task JSONL files."""
    current_idx = 0
    for category, count, output_file in task_info:
        task_results = all_results[current_idx:current_idx + count]
        for line in task_results:
            line.pop('_task_category', None)
            line.pop('_output_file', None)
            line.pop('_ce_scores', None)
            line.pop('_ce_labels', None)
        
        with xopen(output_file, "w", encoding="utf-8") as fout:
            for line in task_results:
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")
        
        print(f"Results for {category} saved to {output_file}")
        current_idx += count

def fast_path_inference(args, models, indices, method_name, all_results, turn1_cache=None):
    """Fast Path: inference on ambiguous / insufficient-context questions.
    
    Always applies logical prompt in Turn 1 and double-check logical reprompting in Turn 2.
    When router_method == 'context_sufficiency', entities are kept masked.
    Turn 1 results can be passed in turn1_cache to avoid duplicate decoding.
    Double-check reprompting allows the model to confirm or correct its answer.
    """
    method = args.method
    router_method = getattr(args, "router_method", "context_sufficiency")
    if router_method == "context_sufficiency":
        is_masked = False  # Consistent samples output unmasked response directly
    else:
        is_masked = getattr(args, "mask_entities", "none") in ("fast_path", "both")
    warnings = "" if is_masked else (UNCERTAINTY_WARNING + DEBIAS_PROMPT)

    # ── Turn 1 (Always logical) ──
    if turn1_cache is not None and all(i in turn1_cache for i in indices):
        prompts_text = [turn1_cache[i][0] for i in indices]
        responses = [turn1_cache[i][1] for i in indices]
    else:
        prompts = []
        prompts_text = []  # raw text for reprompting history
        for i in indices:
            item = mask_entities(all_results[i]) if is_masked else all_results[i]
            prompt_text = get_prompt(args, item, method, warnings=warnings, logical=True, masked=is_masked)
            prompts_text.append(prompt_text)
            prompts.append(format_prompt(args, prompt_text, sys_prompt=DEFAULT_SYS_PROMPT, id=args.summary_id))

        log_data(f"\n{'='*30}\nFast Path Turn 1 (logical, masked={is_masked}):", args)
        responses = decode_batch(models, args, prompts, phase="fast_path_turn1")
        if responses:
            log_data(f"\n{'-'*30}\nFast Path response sample:\n{responses[0]}", args)

    # ── Turn 2: Reprompting (Double-check) only if fast_path_method == 'logical_reprompting' ──
    if getattr(args, "fast_path_method", "logical") == "logical_reprompting":
        reprompt_prompts = []
        fact_ext = getattr(args, "feature_extraction", False)
        reprompt_text = REPROMPT_MASKED if is_masked else (REPROMPT_FACT_EXTRACTION if fact_ext else REPROMPT)
        for j, i in enumerate(indices):
            reprompt_prompt = format_prompt(
                args, reprompt_text + POSTFIX_PROMPT,
                sys_prompt=DEFAULT_SYS_PROMPT,
                thinking_history=[[prompts_text[j], responses[j]]],
                id=args.summary_id,
            )
            reprompt_prompts.append(reprompt_prompt)

        log_data(f"\n{'='*30}\nFast Path Turn 2 Reprompting ({len(reprompt_prompts)} samples, masked={is_masked}):", args)
        resp2 = decode_batch(models, args, reprompt_prompts, phase="fast_path_turn2")
        for j in range(len(indices)):
            if j < len(resp2) and resp2[j]:
                responses[j] = resp2[j]
        if responses:
            log_data(f"\n{'-'*30}\nFast Path Turn 2 response sample:\n{responses[0]}", args)
    else:
        log_data(f"\nFast Path skipping Turn 2 reprompting (method={getattr(args, 'fast_path_method', 'logical')}) to preserve Turn 1 accuracy and avoid Self-Correction Paradox.", args)

    for j, i in enumerate(indices):
        pred = get_pred(responses[j])
        if pred == -1:
            if turn1_cache and i in turn1_cache:
                pred = get_pred(turn1_cache[i][1])
            if pred == -1:
                pred = get_undetermined_index(all_results[i])

        if method_name in all_results[i] and not args.overwrite:
            all_results[i][method_name][0].append(pred)
            all_results[i][method_name][1].append([])
            all_results[i][method_name][2].append(responses[j])
        else:
            all_results[i][method_name] = [[pred], [[]], [responses[j]]]

def adaptive_path_inference(args, models, indices, method_name, all_results, routes):
    """Adaptive Persona Path: multi-round debate with personas.
    
    Supports:
    - 'symmetric': standard MPT multi-round debate with top-k personas.
    - 'asymmetric_tribunal': 3-Persona Tribunal (Fact-Anchor, Bias-Auditor, Judge Persona).
      In each round, the 3 personas generate their reasoning.
      In the final round, the Judge Persona produces the final <answer> </answer> verdict.
    """
    method = args.method
    inference_type = args.inference_type
    N = len(indices)
    debate_mode = getattr(args, "debate_mode", "symmetric")

    router_method = getattr(args, "router_method", "context_sufficiency")
    if router_method == "context_sufficiency":
        is_masked = False  # Always unmask in Adaptive Path
    else:
        is_masked = getattr(args, "mask_entities", "none") == "both"

    # ── Build per-sample personas and warnings ──
    per_sample_personas = []
    per_sample_warnings = []

    if debate_mode == "asymmetric_tribunal":
        tribunal_personas = [
            FACT_ANCHOR_SYS_PROMPT,
            BIAS_AUDITOR_SYS_PROMPT,
            JUDGE_PERSONA_SYS_PROMPT,
        ]
        for _ in range(N):
            per_sample_personas.append(tribunal_personas)
            per_sample_warnings.append("")
    else:
        for idx_pos, i in enumerate(indices):
            if is_masked:
                personas = [PERSONA_SYS_PROMPT.format(persona="impartial observer")]
                warn = ""
            elif args.disable_cross_encoder or routes is None:
                groups = get_groups(args, all_results[i])
                personas = [PERSONA_SYS_PROMPT.format(persona=g) for g in groups]
                personas.insert(0, PERSONA_SYS_PROMPT.format(persona="neutral general public"))
                warn = DEBIAS_PROMPT
            else:
                if isinstance(routes, dict):
                    bias_labels = routes[i][1] if i in routes else []
                elif len(routes) == N:
                    bias_labels = routes[idx_pos][1]
                elif len(routes) > i:
                    bias_labels = routes[i][1]
                else:
                    bias_labels = []

                persona_descs = get_personas_from_labels(bias_labels)
                personas = [PERSONA_SYS_PROMPT.format(persona=p) for p in persona_descs]
                bias_types = ", ".join(bias_labels)
                warn = BIAS_WARNING.format(bias_types=bias_types) + DEBIAS_PROMPT
                personas.insert(0, PERSONA_SYS_PROMPT.format(persona="neutral general public"))

            per_sample_personas.append(personas)
            per_sample_warnings.append(warn)

    max_agents = max(len(p) for p in per_sample_personas)
    pred_history = [[[] for _ in range(args.num_rounds + 1)] for _ in range(N)]
    debate_history = [[[] for _ in range(args.num_rounds + 1)] for _ in range(N)]

    # ── Multi-round debate ──
    for round in range(args.num_rounds + 1):
        round_history = [[] for _ in range(N)]

        for agent_id in range(max_agents):
            formatted_prompts = []
            valid_map = []

            for j in range(N):
                if agent_id >= len(per_sample_personas[j]):
                    continue
                i = indices[j]
                is_init = (round == 0)
                item = mask_entities(all_results[i]) if is_masked else all_results[i]
                prompt = get_prompt(args, item, method,
                                    is_init=is_init, warnings=per_sample_warnings[j] if is_init else "",
                                    logical=is_init, masked=is_masked)

                if len(debate_history[j][round]) == 0:
                    debate_history[j][round].append(prompt)

                formatted_prompts.append(format_prompt(
                    args, prompt,
                    sys_prompt=per_sample_personas[j][agent_id],
                    thinking_history=debate_history[j],
                    id=agent_id,
                ))
                valid_map.append(j)

            if formatted_prompts:
                log_data(f"\n{'='*30}\nAdaptive Path ({debate_mode}), Round {round}, Agent {agent_id}:", args)
                responses = decode_batch(models, args, formatted_prompts, phase=f"adaptive_debate_r{round}_a{agent_id}")
                if responses:
                    log_data(f"\n{'-'*30}\nResponse sample:\n{responses[0]}", args)

                for resp_idx, j in enumerate(valid_map):
                    resp = responses[resp_idx]
                    pred = get_pred(resp)
                    pred_history[j][round].append([pred])
                    round_history[j].append(resp)

        for j in range(N):
            debate_history[j][round].extend(round_history[j])

    # ── Final summary / verdict inference ──
    # In the final round, the Judge Persona produces the final <answer> </answer> verdict.
    final_prompts = []
    final_sys_prompt = JUDGE_PERSONA_SYS_PROMPT if debate_mode == "asymmetric_tribunal" else DEFAULT_SYS_PROMPT
    for j in range(N):
        i = indices[j]
        item = mask_entities(all_results[i]) if is_masked else all_results[i]
        prompt = get_prompt(args, item, method, is_init=False, is_summary=True, masked=is_masked)
        final_prompts.append(format_prompt(
            args, prompt,
            sys_prompt=final_sys_prompt,
            thinking_history=debate_history[j],
            id=args.summary_id,
        ))

    log_data(f"\n{'='*30}\nAdaptive Path ({debate_mode}), Final summary:", args)
    final_responses = decode_batch(models, args, final_prompts, phase="adaptive_summary")
    if final_responses:
        log_data(f"\n{'-'*30}\nFinal response sample:\n{final_responses[0]}", args)

    # ── Write results (same format as MPT) ──
    for j in range(N):
        i = indices[j]
        pred = get_pred(final_responses[j])
        if method_name in all_results[i] and not args.overwrite:
            all_results[i][method_name][0].append(pred)
            all_results[i][method_name][1].append(pred_history[j])
            all_results[i][method_name][2].append(final_responses[j])
        else:
            all_results[i][method_name] = [[pred], [pred_history[j]], [final_responses[j]]]

def batch_inference(args, models, method_name):
    method = args.method
    dataset = args.dataset
    router_method = getattr(args, "router_method", "context_sufficiency")

    if args.split: 
        path_groups = [[path] for path in args.input_paths]
    else:
        path_groups = [args.input_paths]

    for batch_input_paths in path_groups:
        all_results, task_info = load_batch_data(args, batch_input_paths)
        total_N = len(all_results)

        print(f"\nMethod: {method}")
        print(f"\nTotal samples in current batch: {total_N}")
        for category, count, _ in task_info:
            print(f"Task: {category}, Batch samples: {count}")

        turn1_cache = None
        routes = None

        if router_method == "context_sufficiency":
            # ── Phase 1: Dual-Pass Counterfactual Batch Inference (Masked & Unmasked 2N Batch) ──
            print(f"\n--- Phase 1: Dual-Pass Counterfactual Routing (Masked & Unmasked Turn 1) ---")
            log_data(f"\n--- Phase 1: Dual-Pass Counterfactual Routing (Masked & Unmasked Turn 1) ---", args)

            masked_items = [mask_entities(it) for it in all_results]
            turn1_prompts_m = []
            turn1_prompts_u = []
            turn1_texts_m = []
            turn1_texts_u = []

            for it_m, it_u in zip(masked_items, all_results):
                # Pass 1: Masked with LOGICAL_PROMPT_MASKED
                p_text_m = get_prompt(args, it_m, method, warnings="", logical=True, masked=True)
                turn1_texts_m.append(p_text_m)
                turn1_prompts_m.append(format_prompt(args, p_text_m, sys_prompt=DEFAULT_SYS_PROMPT, id=args.summary_id))

                # Pass 2: Unmasked with LOGICAL_PROMPT_ASYMMETRIC_UNMASKED (or LOGICAL_PROMPT)
                p_text_u = get_prompt(args, it_u, method, warnings="", logical=True, masked=False)
                turn1_texts_u.append(p_text_u)
                turn1_prompts_u.append(format_prompt(args, p_text_u, sys_prompt=DEFAULT_SYS_PROMPT, id=args.summary_id))

            # Batch decode all 2N prompts in single unified batch for maximum GPU throughput
            all_turn1_prompts = turn1_prompts_m + turn1_prompts_u
            all_turn1_responses = decode_batch(models, args, all_turn1_prompts, phase="dual_pass_routing")
            turn1_responses_m = all_turn1_responses[:total_N]
            turn1_responses_u = all_turn1_responses[total_N:]

            turn1_cache = {}
            fast_indices = []
            adapt_indices = []

            for i in range(total_N):
                resp_m = turn1_responses_m[i] if i < len(turn1_responses_m) else ""
                resp_u = turn1_responses_u[i] if i < len(turn1_responses_u) else ""
                pred_m = get_pred(resp_m)
                pred_u = get_pred(resp_u)

                # Store the unmasked response in turn1_cache for Fast Path direct output
                turn1_cache[i] = (turn1_texts_u[i], resp_u)

                # Consistency Gate with Over-refusal Guard (100% Zero-Oracle):
                # 1. Consistent factual entity (pred_m == pred_u != undetermined): Safe for Fast Path.
                # 2. Consistent undetermined (pred_m == pred_u == undetermined):
                #    Detect internal contradiction (Fact-Anchor cited text evidence in unmasked CoT
                #    despite arbiter choosing undetermined) -> Route to Tribunal to resolve.
                #    Otherwise Safe for Fast Path.
                u_idx = get_undetermined_index(all_results[i])

                if pred_m == pred_u and pred_m in (0, 1, 2):
                    is_over_refusal = False
                    if pred_m == u_idx:
                        # Zero-Oracle: check whether unmasked CoT extracted factual quotes or named actions
                        resp_u_lower = resp_u.lower()
                        has_fact_quote = any(kw in resp_u_lower for kw in [
                            "explicitly states", "verbatim quote", "explicit facts",
                            "fact-anchor: the text", "states that the", "according to the text",
                            "evidence shows that", "the text directly names"
                        ])
                        is_over_refusal = has_fact_quote

                    if is_over_refusal:
                        adapt_indices.append(i)
                    else:
                        fast_indices.append(i)
                else:
                    adapt_indices.append(i)

            print(f"\nDual-Pass Routing: {len(fast_indices)} ({len(fast_indices)/total_N*100:.1f}%) → Fast Path (Consistent, Immune to Bias), {len(adapt_indices)} ({len(adapt_indices)/total_N*100:.1f}%) → Adaptive Tribunal Path (Discrepant/Over-refused, Needs Debate)")
            log_data(f"Dual-Pass Routing: {len(fast_indices)} Fast Path, {len(adapt_indices)} Adaptive Path", args)

            # ── Phase 2: Run Cross-Encoder ONCE only on Adaptive Path samples to extract bias types ──
            # In Asymmetric Tribunal mode, personas are structural and fixed (Fact-Anchor, Bias-Auditor, Judge), so Cross-Encoder is skipped.
            if (
                adapt_indices
                and getattr(args, "debate_mode", "symmetric") != "asymmetric_tribunal"
                and not args.disable_cross_encoder
                and hasattr(args, "_router")
                and args._router is not None
            ):
                print(f"Running Cross-Encoder once on {len(adapt_indices)} Adaptive Path samples to extract bias personas & warnings...")
                router = args._router
                adapt_premises = [f"{all_results[i]['context']} {all_results[i]['question']}" for i in adapt_indices]
                adapt_routes = router.route_batch(adapt_premises, args.top_k, args.top_c)
                routes = {}
                for pos, i in enumerate(adapt_indices):
                    _, labels, scores = adapt_routes[pos]
                    routes[i] = (False, labels, scores)
                    all_results[i]["_ce_scores"] = scores
                    all_results[i]["_ce_labels"] = labels
            else:
                routes = None

        elif router_method == "cross_encoder":
            # Legacy Cross-Encoder Upfront Routing
            if not args.disable_cross_encoder and hasattr(args, "_router") and args._router is not None:
                router = args._router
                premises = [f"{line['context']} {line['question']}" for line in all_results]
                raw_routes = router.route_batch(premises, args.top_k, args.top_c)
                fast_indices = [i for i, (is_fast, _, _) in enumerate(raw_routes) if is_fast]
                adapt_indices = [i for i, (is_fast, _, _) in enumerate(raw_routes) if not is_fast]
                routes = {i: raw_routes[i] for i in range(total_N)}
                for i in range(total_N):
                    all_results[i]["_ce_scores"] = raw_routes[i][2]
                    all_results[i]["_ce_labels"] = raw_routes[i][1]
                print(f"\nRouting (Cross-Encoder): {len(fast_indices)} → Fast Path, {len(adapt_indices)} → Adaptive Path")
                log_data(f"Routing: {len(fast_indices)} Fast Path, {len(adapt_indices)} Adaptive Path", args)
            else:
                routes = None
                fast_indices = []
                adapt_indices = list(range(total_N))
                print(f"\nCross-Encoder disabled, all {total_N} samples → Adaptive Path (legacy)")

        # ── Phase 3: Fast Path ──
        if fast_indices:
            print(f"\n--- Fast Path ({args.fast_path_method}, {len(fast_indices)} samples) ---")
            fast_path_inference(args, models, fast_indices, method_name, all_results, turn1_cache=turn1_cache)

        # ── Phase 4: Adaptive Persona Path ──
        if adapt_indices:
            print(f"\n--- Adaptive Persona Path ({len(adapt_indices)} samples) ---")
            adaptive_path_inference(args, models, adapt_indices, method_name, all_results, routes)

        # ── Phase 5: Write output ──
        write_results(all_results, task_info)

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    method_name = f"{args.method}_{args.num_rounds}_{args.max_tokens}"
    log_start(method_name, args)
    log_data(str(args), args)
    print(f"Inference by: {args.inference_type}")

    # Initialize Cross-Encoder router (only needed for legacy cross_encoder routing or symmetric debate)
    args._router = None
    needs_cross_encoder = (
        not args.disable_cross_encoder
        and (args.router_method == "cross_encoder" or args.debate_mode == "symmetric")
    )
    if needs_cross_encoder:
        print(f"Loading Cross-Encoder: {args.cross_encoder_model} on {args.cross_encoder_device}")
        router = CrossEncoderRouter(args.cross_encoder_model, args.cross_encoder_device, args.dataset)
        if args.ce_cache_file:
            load_ce_cache(router, args.ce_cache_file)
            print(f"Loaded CE cache: {len(router._cache)} entries from {args.ce_cache_file}")
        args._router = router

    models = [Decoder(args, model) for model in args.models]
    batch_inference(args, models, method_name)

    print(f"\nTotal LLM calls: {TOTAL_LLM_CALLS}\n")
    if hasattr(args, "log_path") and args.log_path:
        log_data(f"Total LLM calls: {TOTAL_LLM_CALLS}", args)

    # Save CE cache after inference
    if getattr(args, "_router", None) is not None and args.ce_cache_file:
        save_ce_cache(args._router, args.ce_cache_file)
        print(f"Saved CE cache: {len(args._router._cache)} entries to {args.ce_cache_file}")
            

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method", type=str, default="mpt",
        help="Method"
    )

    parser.add_argument(
        "--note", type=str, default="1",
        help="Notes"
    )

    parser.add_argument(
        "--dataset", type=str, default="BBQ",
        help="Dataset used for experiment"
    )

    parser.add_argument(
        "--models_dir", type=str, default=None,
        help="Models directory"
    )

    parser.add_argument(
        "--inference_type", type=str, default="vllm",
        help="Inference type used to generate responses"
    )

    parser.add_argument(
        "--num_agents", type=int, default=3,
        help="Number of agents"
    )

    parser.add_argument(
        "--models", type=str, nargs="+", default=["meta-llama/Llama-3.1-8B-Instruct"], 
        help="Models used to generate responses"
    )

    parser.add_argument(
        "--summary_id", type=int, default=0,
        help="Id of the model used to generate the summary"
    )

    parser.add_argument(
        "--num_rounds", type=int, default=2,
        help="Number of rounds"
    )

    parser.add_argument(
        "--num_attempts", type=int, default=5,
        help="Number of attempts"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature used to generate responses"
    )

    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Top-p used to generate responses"
    )

    parser.add_argument(
        "--max_tokens", type=int, default=768,
        help="The maximum number of tokens the model output"
    )

    parser.add_argument(
        "--random_seed", type=int, default=0,
        help="Random seed"
    )

    parser.add_argument(
        "--num_gpus", type=int, default=4,
        help="Number of GPUs to use"
    )

    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85,
        help="GPU memory utilization ratio for vLLM"
    )

    parser.add_argument(
        "--max_model_len", type=int, default=32768,
        help="Maximum model context length for vLLM"
    )

    parser.add_argument(
        "--overwrite", action="store_true",
        help="Whether overwrite the results"
    )

    parser.add_argument(
        "--split", action="store_true",
        help="Whether split the dataset"
    )

    parser.add_argument(
        "--no_general", action="store_true",
        help="Whether has a neutral general persona"
    )

    parser.add_argument(
        "--truncate", action="store_true",
        help="Whether truncate the dataset"
    )

    parser.add_argument(
        "--chat_templates", action="store_true",
        help="Whether use chat templates"
    )

    # ── Debate Mode ──
    parser.add_argument(
        "--debate_mode", type=str, default="asymmetric_tribunal",
        choices=["symmetric", "asymmetric_tribunal"],
        help="Debate mode in Adaptive Path: 'asymmetric_tribunal' (Fact-Anchor, Bias-Auditor, Judge) or 'symmetric'"
    )

    parser.add_argument(
        "--asymmetric", action="store_true", default=None,
        help="Enable Asymmetric mode (automatically sets --debate_mode asymmetric_tribunal)"
    )

    # ── Routing Strategy ──
    parser.add_argument(
        "--router_method", type=str, default="context_sufficiency",
        choices=["context_sufficiency", "cross_encoder"],
        help="Routing strategy: 'context_sufficiency' (masked LLM reasoning) or 'cross_encoder' (NLI logits)"
    )

    # ── Cross-Encoder Routing Args ──
    parser.add_argument(
        "--cross_encoder_model", type=str, default="cross-encoder/nli-distilroberta-base",
        help="HuggingFace model for Cross-Encoder NLI scoring"
    )

    parser.add_argument(
        "--cross_encoder_device", type=str, default="cpu",
        help="Device for Cross-Encoder model (cpu, cuda, cuda:0, etc.)"
    )

    parser.add_argument(
        "--top_k", type=int, default=2,
        help="Top-k bias categories to activate as personas in Adaptive Path"
    )

    parser.add_argument(
        "--top_c", type=int, default=2,
        help="If 'neutral' ranks within top-c, route sample to Fast Path"
    )

    parser.add_argument(
        "--fast_path_method", type=str, default="logical",
        choices=["direct", "logical", "logical_reprompting"],
        help="Prompting strategy for Fast Path: direct, logical (no reprompting), or logical_reprompting"
    )

    parser.add_argument(
        "--ce_cache_file", type=str, default=None,
        help="Path to save/load Cross-Encoder score cache (JSONL)"
    )

    parser.add_argument(
        "--disable_cross_encoder", action="store_true",
        help="Disable Cross-Encoder routing, use legacy get_groups() for all samples"
    )

    parser.add_argument(
        "--mask_entities", type=str, default="none",
        choices=["none", "fast_path", "both"],
        help="Entity masking mode: none (disabled), fast_path (option 1: Fast Path only), or both (option 2: both paths)"
    )

    parser.add_argument(
        "--feature_extraction", "--fact_extraction", action="store_true",
        help="Use extractive evidence grounding prompts ('Quote-First' fact extraction)"
    )

    args = parser.parse_args()

    args.fact_extraction = args.feature_extraction

    # Synchronize --asymmetric and --debate_mode
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
    
    if args.dataset == "stereoset":
        tasks = set_tasks_stereoset()
        args.input_paths = [f"./data/{args.dataset}/{task}.jsonl" for task in tasks]
    
    if args.method.startswith("m"):
        if not args.method.endswith("sc"):
            args.num_attempts = 1

    return args


if __name__ == "__main__":
    main()
