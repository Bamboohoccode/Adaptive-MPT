import re
from adaptive_mpt.prompts import (
    DEFAULT_SYS_PROMPT,
    QUESTION_PROMPT_BBQ,
    QUESTION_PROMPT_STEREOSET,
    CONSTRAINT_PROMPT_BBQ,
    CONSTRAINT_PROMPT_STS,
    DEBIAS_PROMPT,
    UNCERTAINTY_WARNING,
    POSTFIX_PROMPT,
    LOGICAL_PROMPT,
    REPROMPT,
    LOGICAL_PROMPT_MASKED,
    REPROMPT_MASKED,
    REVIEW_PROMPT_MASKED,
    SUMMARY_PROMPT_MASKED,
    LOGICAL_PROMPT_FACT_EXTRACTION,
    REPROMPT_FACT_EXTRACTION,
    REVIEW_PROMPT_FACT_EXTRACTION,
    SUMMARY_PROMPT_FACT_EXTRACTION,
    LOGICAL_PROMPT_ASYMMETRIC_UNMASKED,
    FACT_ANCHOR_SYS_PROMPT,
    BIAS_AUDITOR_SYS_PROMPT,
    JUDGE_PERSONA_SYS_PROMPT,
)
from adaptive_mpt.data import mask_entities, get_undetermined_index
from adaptive_mpt.models import format_prompt, decode_batch, get_pred

def build_prompt(args, input_example, is_init=True, warnings="", logical=False, is_summary=False, masked=False, is_reprompt=False):
    """Construct prompt text according to inference phase and entity masking state."""
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

def run_dual_pass_screening(args, models, sample_records):
    """Component 1: Dual-Pass Counterfactual Batch Inference (Unified 2N Batch).
    
    Pass 1 (Masked Logic): [Person A], [Person B] with LOGICAL_PROMPT_MASKED -> P_M
    Pass 2 (Unmasked CoT): Original identities with LOGICAL_PROMPT_ASYMMETRIC_UNMASKED -> P_U, R_U
    """
    total_samples = len(sample_records)
    print("\n--- Phase 1: Dual-Pass Counterfactual Screening (Masked & Unmasked 2N Batch) ---")

    counterfactual_masked_inputs = [mask_entities(it) for it in sample_records]

    masked_prompts = []
    unmasked_prompts = []
    for i in range(total_samples):
        p_m = build_prompt(args, counterfactual_masked_inputs[i], is_init=True, logical=True, masked=True)
        masked_prompts.append(format_prompt(args, p_m, sys_prompt=DEFAULT_SYS_PROMPT))

        p_u = build_prompt(args, sample_records[i], is_init=True, logical=True, masked=False)
        unmasked_prompts.append(format_prompt(args, p_u, sys_prompt=DEFAULT_SYS_PROMPT))

    # Single unified 2N batch execution on vLLM
    batch_prompts = masked_prompts + unmasked_prompts
    batch_responses = decode_batch(models, args, batch_prompts, phase="dual_pass_batch_turn1")

    masked_responses = batch_responses[:total_samples]
    unmasked_responses = batch_responses[total_samples:]

    p_masked = []
    p_unmasked = []
    cot_unmasked = []
    fast_path_cache = {}

    for i in range(total_samples):
        pm = get_pred(masked_responses[i])
        pu = get_pred(unmasked_responses[i])
        p_masked.append(pm)
        p_unmasked.append(pu)
        cot_unmasked.append(unmasked_responses[i])
        # Cache Turn 1 unmasked response for 0-compute Fast Path
        fast_path_cache[i] = (unmasked_prompts[i], unmasked_responses[i])

    return p_masked, p_unmasked, cot_unmasked, fast_path_cache

def route_samples(args, sample_records, p_masked, p_unmasked, cot_unmasked):
    """Component 2: Zero-Oracle Consistency Gate with Over-refusal Guard."""
    total_samples = len(sample_records)
    fast_path_indices = []
    tribunal_indices = []

    for i in range(total_samples):
        pm = p_masked[i]
        pu = p_unmasked[i]
        u_idx = get_undetermined_index(sample_records[i])

        if pm == pu:
            is_over_refusal = False
            if pm == u_idx and not args.truncate:
                cot_lower = cot_unmasked[i].lower()
                factual_evidence_anchored = any(kw in cot_lower for kw in [
                    "the context states", "the context explicitly", "according to the context",
                    "the text states", "as stated in the", "explicitly mentions",
                    "directly mentions", "explicitly states", "verbatim quote", "explicit facts",
                    "fact-anchor: the text", "states that the", "according to the text",
                    "evidence shows that", "the text directly names"
                ])
                is_over_refusal = factual_evidence_anchored

            if is_over_refusal:
                tribunal_indices.append(i)
            else:
                fast_path_indices.append(i)
        else:
            tribunal_indices.append(i)

    print(f"\nZero-Oracle Consistency Gate: {len(fast_path_indices)} ({len(fast_path_indices)/total_samples*100:.1f}%) → Fast Path, {len(tribunal_indices)} ({len(tribunal_indices)/total_samples*100:.1f}%) → Asymmetric Tribunal Path")
    return fast_path_indices, tribunal_indices

def execute_fast_path(args, models, fast_path_indices, sample_records, fast_path_cache, experiment_signature):
    """Component 3: Fast Path (System 1) - Direct Turn 1 Cache Output (0 Compute Overhead)."""
    if not fast_path_indices:
        return

    print(f"\n--- Fast Path ({args.fast_path_method}, {len(fast_path_indices)} samples) ---")
    for i in fast_path_indices:
        _, response = fast_path_cache[i]
        pred = get_pred(response)
        if experiment_signature in sample_records[i] and not args.overwrite:
            sample_records[i][experiment_signature][0].append(pred)
            sample_records[i][experiment_signature][1].append([[pred]])
            sample_records[i][experiment_signature][2].append(response)
        else:
            sample_records[i][experiment_signature] = [[pred], [[[pred]]], [response]]

def execute_asymmetric_tribunal(args, models, tribunal_indices, sample_records, experiment_signature):
    """Component 4: Asymmetric Persona Tribunal (System 2: Fact-Anchor, Bias-Auditor, Judge)."""
    if not tribunal_indices:
        return

    num_samples = len(tribunal_indices)
    print(f"\n--- Asymmetric Persona Tribunal ({num_samples} samples) ---")

    tribunal_personas = [
        FACT_ANCHOR_SYS_PROMPT,
        BIAS_AUDITOR_SYS_PROMPT,
        JUDGE_PERSONA_SYS_PROMPT,
    ]
    max_agents = len(tribunal_personas)

    tribunal_vote_trace = [[[] for _ in range(args.num_rounds + 1)] for _ in range(num_samples)]
    tribunal_transcript = [[[] for _ in range(args.num_rounds + 1)] for _ in range(num_samples)]

    # Multi-round adversarial debate
    for round_idx in range(args.num_rounds + 1):
        round_responses = [[] for _ in range(num_samples)]

        for agent_id in range(max_agents):
            formatted_prompts = []
            for j in range(num_samples):
                i = tribunal_indices[j]
                is_init = (round_idx == 0)
                item = sample_records[i]
                prompt = build_prompt(args, item, is_init=is_init, warnings="", logical=is_init, masked=False)

                if len(tribunal_transcript[j][round_idx]) == 0:
                    tribunal_transcript[j][round_idx].append(prompt)

                formatted_prompts.append(format_prompt(
                    args, prompt,
                    sys_prompt=tribunal_personas[agent_id],
                    thinking_history=tribunal_transcript[j],
                    id=agent_id,
                ))

            if formatted_prompts:
                responses = decode_batch(models, args, formatted_prompts, phase=f"tribunal_r{round_idx}_a{agent_id}")
                for j in range(num_samples):
                    resp = responses[j]
                    pred = get_pred(resp)
                    tribunal_vote_trace[j][round_idx].append([pred])
                    round_responses[j].append(resp)

        for j in range(num_samples):
            tribunal_transcript[j][round_idx].extend(round_responses[j])

    # Judicial Arbiter final synthesis verdict
    judge_synthesis_prompts = []
    judge_system_prompt = JUDGE_PERSONA_SYS_PROMPT
    for j in range(num_samples):
        i = tribunal_indices[j]
        item = sample_records[i]
        prompt = build_prompt(args, item, is_init=False, is_summary=True, masked=False)
        judge_synthesis_prompts.append(format_prompt(
            args, prompt,
            sys_prompt=judge_system_prompt,
            thinking_history=tribunal_transcript[j],
            id=args.summary_id,
        ))

    judge_verdicts = decode_batch(models, args, judge_synthesis_prompts, phase="tribunal_judge_summary")

    # Save tribunal predictions
    for j in range(num_samples):
        i = tribunal_indices[j]
        pred = get_pred(judge_verdicts[j])
        if experiment_signature in sample_records[i] and not args.overwrite:
            sample_records[i][experiment_signature][0].append(pred)
            sample_records[i][experiment_signature][1].append(tribunal_vote_trace[j])
            sample_records[i][experiment_signature][2].append(judge_verdicts[j])
        else:
            sample_records[i][experiment_signature] = [[pred], [tribunal_vote_trace[j]], [judge_verdicts[j]]]
