import os
import re
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI

from adaptive_mpt.prompts import (
    BEG_LLAMA3, SYS_LLAMA3, USR_LLAMA3, AST_LLAMA3, END_LLAMA3,
    SYS_QWEN, USR_QWEN, AST_QWEN, END_QWEN,
)

TOTAL_LLM_CALLS = 0

def get_total_llm_calls():
    """Return total number of prompt decodings executed."""
    return TOTAL_LLM_CALLS

def reset_total_llm_calls():
    """Reset LLM call counter."""
    global TOTAL_LLM_CALLS
    TOTAL_LLM_CALLS = 0

class LLMDecoder:
    """Production LLM Decoder wrapping vLLM and OpenAI APIs ."""
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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.inference_type == "vllm":
            num_gpus = getattr(args, "num_gpus", 1)
            gpu_mem = getattr(args, "gpu_memory_utilization", 0.85)
            max_len = getattr(args, "max_model_len", 32768)
            self.model = LLM(
                model=self.model_name,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=gpu_mem,
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
            outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=True)
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
                return outputs
            except Exception as e:
                print(f"\nRetrying due to API error: {e}\n")
                time.sleep(5)
                return self.decode(prompt)

def format_prompt(args, usr_prompt, sys_prompt=None, thinking_history=None, id=0):
    inference_type = args.inference_type
    is_chat_templates = args.chat_templates
    model_name = args.models[id] if inference_type == "openai" else args.models[0]

    if inference_type == "vllm" and not is_chat_templates:
        formatted_prompt = [BEG_LLAMA3] if model_name.startswith("meta-llama") else []
    else:
        formatted_prompt = []

    # System prompt
    if inference_type == "vllm" and not is_chat_templates:
        if model_name.startswith("meta-llama"):
            formatted_prompt.extend([SYS_LLAMA3, sys_prompt, END_LLAMA3])
        elif model_name.startswith("Qwen"):
            formatted_prompt.extend([SYS_QWEN, sys_prompt, END_QWEN])
    else:
        formatted_prompt.extend([{"role": "system", "content": sys_prompt}])

    # History
    if thinking_history:
        for round_history in thinking_history:
            if len(round_history) > 1:
                init_prompt = round_history[0]
                if inference_type == "vllm" and not is_chat_templates:
                    if model_name.startswith("meta-llama"):
                        formatted_prompt.extend([USR_LLAMA3, init_prompt, END_LLAMA3, AST_LLAMA3])
                    elif model_name.startswith("Qwen"):
                        formatted_prompt.extend([USR_QWEN, init_prompt, END_QWEN, AST_QWEN])
                else:
                    formatted_prompt.extend([{"role": "user", "content": init_prompt}])

                history = [f"Response {i}: \n{{{' '.join(resp.splitlines())}}}" for i, resp in enumerate(round_history[1:])]
                if inference_type == "vllm" and not is_chat_templates:
                    formatted_prompt.extend(["\n".join(history), END_LLAMA3 if model_name.startswith("meta-llama") else END_QWEN])
                else:
                    formatted_prompt.extend([{"role": "assistant", "content": "\n".join(history)}])

    # Current prompt
    if inference_type == "vllm" and not is_chat_templates:
        if model_name.startswith("meta-llama"):
            formatted_prompt.extend([USR_LLAMA3, usr_prompt, END_LLAMA3, AST_LLAMA3])
        elif model_name.startswith("Qwen"):
            formatted_prompt.extend([USR_QWEN, usr_prompt, END_QWEN, AST_QWEN])
        return "\n".join(formatted_prompt)
    else:
        formatted_prompt.extend([{"role": "user", "content": usr_prompt}])
        return formatted_prompt

def decode_batch(models, args, formatted_prompts, **kwargs):
    """Unified decode for both vllm and openai backends."""
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

def get_pred(resp):
    """Robust extraction of answer choices (0, 1, 2 or -1)."""
    try:
        tag_matches = re.findall(r'<answer>(.*?)</answer>', resp, re.IGNORECASE | re.DOTALL)
        if tag_matches:
            inner_text = tag_matches[-1].strip()
            m = re.search(r'([0-2])', inner_text)
            if m:
                return int(m.group(1))
            m_letter = re.search(r'\b([A-C])\b', inner_text, re.IGNORECASE)
            if m_letter:
                return ord(m_letter.group(1).upper()) - ord('A')

        end_snippet = resp[-200:]
        m_decl = re.search(r'(?:final answer|correct option|choice)[:\s*]+([0-2]|[A-C])\b', end_snippet, re.IGNORECASE)
        if m_decl:
            val = m_decl.group(1)
            return int(val) if val.isdigit() else (ord(val.upper()) - ord('A'))

        m = re.search(r'\b([0-2])\b', end_snippet)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1
