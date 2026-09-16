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
from collections import Counter
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts import (
    BEG_LLAMA3, SYS_LLAMA3, USR_LLAMA3, AST_LLAMA3, END_LLAMA3,
    SYS_QWEN, USR_QWEN, AST_QWEN, END_QWEN,
    DEFAULT_SYS_PROMPT, PERSONA_SYS_PROMPT,
    QUESTION_PROMPT_BBQ, QUESTION_PROMPT_STEREOSET,
    DEBIAS_PROMPT, PERSONA_PROMPT, REVIEW_PROMPT, POSTFIX_PROMPT,
    CONSTRAINT_PROMPT_BBQ, CONSTRAINT_PROMPT_STS,
    UNCERTAINTY_WARNING, BIAS_WARNING, LOGICAL_PROMPT, REPROMPT,
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

        if self.is_chat_templates:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                # cache_dir=args.models_dir
                )

        if self.inference_type == "vllm":
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=args.num_gpus,
                gpu_memory_utilization=0.85,
                # download_dir=args.models_dir,
                max_model_len=32768,
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
        if self.inference_type == "vllm":
            if self.is_chat_templates:
                prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

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
    with open(args.log_path + '/loggings.txt', 'a', encoding='utf-8') as f:
        f.write(str(text))
        # print(text)
        f.write('\n')

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

def get_prompt(args, input_example, method, is_init=True, warnings="", logical=False):
    dataset = args.dataset

    context = input_example["context"]
    question = input_example["question"]
    options = f"Options: (0): {input_example['ans0']}; (1): {input_example['ans1']}; (2): {input_example['ans2']}"

    if dataset == "BBQ":
        ask_prompt = CONSTRAINT_PROMPT_BBQ.format(question_prompt=QUESTION_PROMPT_BBQ, context=context, question=question, options=options)
    elif dataset == "stereoset":
        ask_prompt = CONSTRAINT_PROMPT_STS.format(question_prompt=QUESTION_PROMPT_STEREOSET, context=context, question=question, options=options)

    if is_init:
        prompt = ask_prompt + warnings
        if logical:
            prompt += LOGICAL_PROMPT
    else:
        prompt = REVIEW_PROMPT

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
        tag_match = re.search(r'<answer>(.*?)</answer>', resp, re.IGNORECASE | re.DOTALL)
        
        if tag_match:
            inner_text = tag_match.group(1)
            m = re.search(r'\b([0-2])\b', inner_text)
            if m:
                return int(m.group(1))
            
        m = re.search(r'\b([0-2])\b', resp)
        return int(m.group(1)) if m else -1
        
    except Exception:
        return -1

def decode_batch(models, args, formatted_prompts):
    """Unified decode for both vllm and openai backends."""
    if not formatted_prompts:
        return []
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

def fast_path_inference(args, models, indices, method_name, all_results):
    """Fast Path: single-turn inference with UNCERTAINTY_WARNING + DEBIAS_PROMPT.
    
    Supports 3 strategies: direct, logical, logical_reprompting.
    """
    method = args.method
    is_logical = args.fast_path_method in ("logical", "logical_reprompting")
    warnings = UNCERTAINTY_WARNING + DEBIAS_PROMPT

    # ── Turn 1 ──
    prompts = []
    prompts_text = []  # raw text for reprompting history
    for i in indices:
        prompt_text = get_prompt(args, all_results[i], method, warnings=warnings, logical=is_logical)
        prompts_text.append(prompt_text)
        prompts.append(format_prompt(args, prompt_text, sys_prompt=DEFAULT_SYS_PROMPT, id=args.summary_id))

    log_data(f"\n{'='*30}\nFast Path Turn 1 ({args.fast_path_method}):", args)
    responses = decode_batch(models, args, prompts)
    if responses:
        log_data(f"\n{'-'*30}\nFast Path response sample:\n{responses[0]}", args)

    # ── Turn 2: Reprompting (logical_reprompting only) ──
    if args.fast_path_method == "logical_reprompting":
        for j, i in enumerate(indices):
            pred = get_pred(responses[j])
            # Reprompt if parse failure OR pred matches "unknown" label
            needs_reprompt = (pred == -1) or (pred == all_results[i].get("label", -99))
            if needs_reprompt:
                reprompt_prompt = format_prompt(
                    args, REPROMPT + POSTFIX_PROMPT,
                    sys_prompt=DEFAULT_SYS_PROMPT,
                    thinking_history=[[prompts_text[j], responses[j]]],
                    id=args.summary_id,
                )
                log_data(f"\n{'-'*30}\nFast Path reprompt for sample {i}", args)
                resp2 = decode_batch(models, args, [reprompt_prompt])
                if resp2:
                    responses[j] = resp2[0]

    # ── Write results (same format as MPT) ──
    for j, i in enumerate(indices):
        pred = get_pred(responses[j])
        if method_name in all_results[i] and not args.overwrite:
            all_results[i][method_name][0].append(pred)
            all_results[i][method_name][1].append([])
            all_results[i][method_name][2].append(responses[j])
        else:
            all_results[i][method_name] = [[pred], [[]], [responses[j]]]

def adaptive_path_inference(args, models, indices, method_name, all_results, routes):
    """Adaptive Persona Path: multi-round debate with top-k personas + BIAS_WARNING + DEBIAS_PROMPT."""
    method = args.method
    inference_type = args.inference_type
    N = len(indices)

    # ── Build per-sample personas and warnings ──
    per_sample_personas = []
    per_sample_warnings = []
    for idx_pos, i in enumerate(indices):
        if args.disable_cross_encoder:
            groups = get_groups(args, all_results[i])
            personas = [PERSONA_SYS_PROMPT.format(persona=g) for g in groups]
            warn = DEBIAS_PROMPT
        else:
            bias_labels = routes[i][1]
            persona_descs = get_personas_from_labels(bias_labels)
            personas = [PERSONA_SYS_PROMPT.format(persona=p) for p in persona_descs]
            bias_types = ", ".join(bias_labels)
            warn = BIAS_WARNING.format(bias_types=bias_types) + DEBIAS_PROMPT

        # Always prepend neutral general persona
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
                prompt = get_prompt(args, all_results[i], method,
                                    is_init=is_init, warnings=per_sample_warnings[j] if is_init else "")

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
                log_data(f"\n{'='*30}\nAdaptive Path, Round {round}, Agent {agent_id}:", args)
                responses = decode_batch(models, args, formatted_prompts)
                if responses:
                    log_data(f"\n{'-'*30}\nResponse sample:\n{responses[0]}", args)

                for resp_idx, j in enumerate(valid_map):
                    resp = responses[resp_idx]
                    pred = get_pred(resp)
                    pred_history[j][round].append([pred])
                    round_history[j].append(resp)

        for j in range(N):
            debate_history[j][round].extend(round_history[j])

    # ── Final summary inference ──
    final_prompts = []
    for j in range(N):
        i = indices[j]
        prompt = get_prompt(args, all_results[i], method, is_init=False)
        final_prompts.append(format_prompt(
            args, prompt,
            sys_prompt=DEFAULT_SYS_PROMPT,
            thinking_history=debate_history[j],
            id=args.summary_id,
        ))

    log_data(f"\n{'='*30}\nAdaptive Path, Final summary:", args)
    final_responses = decode_batch(models, args, final_prompts)
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

        # ── Phase 1: Cross-Encoder Routing ──
        if not args.disable_cross_encoder:
            router = args._router
            premises = [f"{line['context']} {line['question']}" for line in all_results]
            routes = router.route_batch(premises, args.top_k, args.top_c)
            fast_indices = [i for i, (is_fast, _, _) in enumerate(routes) if is_fast]
            adapt_indices = [i for i, (is_fast, _, _) in enumerate(routes) if not is_fast]

            # Store routing info for logging
            for i, (_, labels, scores) in enumerate(routes):
                all_results[i]["_ce_scores"] = scores
                all_results[i]["_ce_labels"] = labels

            print(f"\nRouting: {len(fast_indices)} → Fast Path, {len(adapt_indices)} → Adaptive Path")
            log_data(f"Routing: {len(fast_indices)} Fast Path, {len(adapt_indices)} Adaptive Path", args)
        else:
            routes = None
            fast_indices = []
            adapt_indices = list(range(total_N))
            print(f"\nCross-Encoder disabled, all {total_N} samples → Adaptive Path (legacy)")

        # ── Phase 2: Fast Path ──
        if fast_indices:
            print(f"\n--- Fast Path ({args.fast_path_method}) ---")
            fast_path_inference(args, models, fast_indices, method_name, all_results)

        # ── Phase 3: Adaptive Persona Path ──
        if adapt_indices:
            print(f"\n--- Adaptive Persona Path (top-{args.top_k} personas) ---")
            adaptive_path_inference(args, models, adapt_indices, method_name, all_results, routes)

        # ── Phase 4: Write output ──
        write_results(all_results, task_info)

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    method_name = f"{args.method}_{args.num_rounds}_{args.max_tokens}"
    log_start(method_name, args)
    log_data(str(args), args)
    print(f"Inference by: {args.inference_type}")

    # Initialize Cross-Encoder router
    if not args.disable_cross_encoder:
        print(f"Loading Cross-Encoder: {args.cross_encoder_model} on {args.cross_encoder_device}")
        router = CrossEncoderRouter(args.cross_encoder_model, args.cross_encoder_device, args.dataset)
        if args.ce_cache_file:
            load_ce_cache(router, args.ce_cache_file)
            print(f"Loaded CE cache: {len(router._cache)} entries from {args.ce_cache_file}")
        args._router = router

    models = [Decoder(args, model) for model in args.models]
    batch_inference(args, models, method_name)

    # Save CE cache after inference
    if not args.disable_cross_encoder and args.ce_cache_file:
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
        "--num_rounds", type=int, default=3,
        help="Number of rounds"
    )

    parser.add_argument(
        "--num_attempts", type=int, default=5,
        help="Number of attempts"
    )

    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature used to generate responses"
    )

    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Top-p used to generate responses"
    )

    parser.add_argument(
        "--max_tokens", type=int, default=512,
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

    # ── Cross-Encoder Routing Args ──
    parser.add_argument(
        "--cross_encoder_model", type=str, default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
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
        "--fast_path_method", type=str, default="logical_reprompting",
        choices=["direct", "logical", "logical_reprompting"],
        help="Prompting strategy for Fast Path: direct, logical, or logical_reprompting"
    )

    parser.add_argument(
        "--ce_cache_file", type=str, default=None,
        help="Path to save/load Cross-Encoder score cache (JSONL)"
    )

    parser.add_argument(
        "--disable_cross_encoder", action="store_true",
        help="Disable Cross-Encoder routing, use legacy get_groups() for all samples"
    )

    args = parser.parse_args()

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
