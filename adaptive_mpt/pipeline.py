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
    SUMMARY_PROMPT,
    REVIEW_PROMPT
)
from adaptive_mpt.data import mask_entities, swap_entities, get_undetermined_index
from adaptive_mpt.models import format_prompt, format_messages, decode_batch, get_pred

def task_prompt(args, input_example):
    """The question itself: instruction, context, question and options."""
    context = input_example["context"]
    question = input_example["question"]
    options = f"Options: (0): {input_example['ans0']}; (1): {input_example['ans1']}; (2): {input_example['ans2']}"
    if args.dataset == "stereoset":
        return CONSTRAINT_PROMPT_STS.format(question_prompt=QUESTION_PROMPT_STEREOSET, context=context, question=question, options=options)
    return CONSTRAINT_PROMPT_BBQ.format(question_prompt=QUESTION_PROMPT_BBQ, context=context, question=question, options=options)

def build_prompt(args, input_example, is_init=True, warnings="", logical=False, is_summary=False, masked=False, is_reprompt=False):
    """Construct prompt text according to inference phase and entity masking state."""
    ask_prompt = task_prompt(args, input_example)

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

def _norm(text):
    """Lower-case, drop punctuation except placeholder brackets, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\[\] ]", " ", str(text).lower())).strip()

def extract_quote(resp):
    """Text of the last <quote> </quote> block, or None."""
    found = re.findall(r"<quote>(.*?)</quote>", resp, re.IGNORECASE | re.DOTALL)
    return found[-1].strip() if found else None

def valid_quote(resp, item, entity_map, masked=False):
    """True iff the response quotes, word for word, a context sentence that names one of the candidates.

    Candidates are the non-unknown option texts plus the names (or, for a masked item, the placeholders)
    from mask_entities' entity map; option texts alone miss names such as Race_x_SES "Juan Rodriguez physician".
    Datasets without people in the options (StereoSet) never have a deciding sentence, so this is False there.
    """
    if not item.get("answer_info"):
        return False
    quote = extract_quote(resp)
    if not quote or quote.upper() == "NONE":
        return False
    q = _norm(quote)
    u = get_undetermined_index(item)
    names = [_norm(item[f"ans{k}"]).removeprefix("the ") for k in range(3) if k != u]
    names += [_norm(ph if masked else name) for name, ph in entity_map.items()]
    return len(q.split()) >= 4 and q in _norm(item["context"]) and any(n and n in q for n in names)

def run_dual_pass_screening(args, models, sample_records):
    """Component 1: Counterfactual Batch Inference (masked, unmasked and identity-swapped passes in one batch).

    Pass 1 (Masked Logic): [Person A], [Person B] with LOGICAL_PROMPT_MASKED -> P_M
    Pass 2 (Unmasked CoT): Original identities with LOGICAL_PROMPT_ASYMMETRIC_UNMASKED -> P_U, R_U
    Pass 3 (Swapped, unless --no_swap_pass): the two people's identities exchanged -> P_S, expected index_map[P_U]
    Per-sample screening details (predictions, quote checks, raw masked/swapped responses) go to record["_screen"].
    """
    total_samples = len(sample_records)
    swap_enabled = getattr(args, "swap_pass", True)
    print(f"\n--- Phase 1: Counterfactual Screening (masked + unmasked{' + swapped' if swap_enabled else ''}) ---")

    counterfactual_masked_inputs = [mask_entities(it) for it in sample_records]
    swaps = [swap_entities(it) if swap_enabled else (None, None) for it in sample_records]

    masked_prompts = []
    unmasked_prompts = []
    swap_prompts, swap_owners = [], []
    for i in range(total_samples):
        p_m = build_prompt(args, counterfactual_masked_inputs[i], is_init=True, logical=True, masked=True)
        masked_prompts.append(format_prompt(args, p_m, sys_prompt=DEFAULT_SYS_PROMPT))

        p_u = build_prompt(args, sample_records[i], is_init=True, logical=True, masked=False)
        unmasked_prompts.append(format_prompt(args, p_u, sys_prompt=DEFAULT_SYS_PROMPT))

        swapped, _ = swaps[i]
        if swapped is not None:
            p_s = build_prompt(args, swapped, is_init=True, logical=True, masked=False)
            swap_prompts.append(format_prompt(args, p_s, sys_prompt=DEFAULT_SYS_PROMPT))
            swap_owners.append(i)

    # Single unified batch execution on vLLM
    batch_prompts = masked_prompts + unmasked_prompts + swap_prompts
    batch_responses = decode_batch(models, args, batch_prompts, phase="dual_pass_batch_turn1")

    masked_responses = batch_responses[:total_samples]
    unmasked_responses = batch_responses[total_samples:2 * total_samples]
    swap_responses = dict(zip(swap_owners, batch_responses[2 * total_samples:]))

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

        entity_map = counterfactual_masked_inputs[i]["_entity_map"]
        screen = {
            "p_m": pm,
            "p_u": pu,
            "vq_m": valid_quote(masked_responses[i], counterfactual_masked_inputs[i], entity_map, masked=True),
            "vq_u": valid_quote(unmasked_responses[i], sample_records[i], entity_map),
            "quote_m": extract_quote(masked_responses[i]),
            "masked_response": masked_responses[i],
        }
        if i in swap_responses:
            screen["p_s"] = get_pred(swap_responses[i])
            screen["p_s_expected"] = swaps[i][1].get(pu, -1)
            screen["swap_response"] = swap_responses[i]
        sample_records[i]["_screen"] = screen

    return p_masked, p_unmasked, cot_unmasked, fast_path_cache

def _legacy_over_refusal(cot):
    """Keyword over-refusal guard (--cot_guard), used only with --no_quote_gate."""
    cot_lower = cot.lower()
    return any(kw in cot_lower for kw in [
        "the context states", "the context explicitly", "according to the context",
        "the text states", "as stated in the", "explicitly mentions",
        "directly mentions", "explicitly states", "verbatim quote", "explicit facts",
        "fact-anchor: the text", "states that the", "according to the text",
        "evidence shows that", "the text directly names"
    ])

def route_samples(args, sample_records, p_masked, p_unmasked, cot_unmasked):
    """Component 2: Consistency Gate.

    Tribunal when: the masked and unmasked answers differ; either is unparseable; the swapped answer does not
    follow the person; both pick a person but the unmasked pass quoted no real deciding sentence (a shared guess);
    or both pick "unknown" although a pass quoted a real sentence naming a candidate (a likely over-refusal).
    --no_quote_gate restores the old gate (with the optional --cot_guard keyword check).
    """
    total_samples = len(sample_records)
    fast_path_indices = []
    tribunal_indices = []
    reasons = {}

    quote_gate = getattr(args, "quote_gate", True)
    cot_guard_enabled = getattr(args, "cot_guard", False)
    if not quote_gate and cot_guard_enabled:
        print("CoT Over-Refusal Guard: ENABLED")

    for i in range(total_samples):
        pm = p_masked[i]
        pu = p_unmasked[i]
        u_idx = get_undetermined_index(sample_records[i])
        screen = sample_records[i].setdefault("_screen", {})

        if pm != pu:
            reason = "discrepancy"
        elif pu == -1:
            reason = "unparseable"
        elif "p_s" in screen and screen["p_s"] != screen["p_s_expected"]:
            reason = "swap_inconsistent"
        elif quote_gate:
            if pu != u_idx and not screen.get("vq_u"):
                reason = "person_without_quote"
            elif pu == u_idx and (screen.get("vq_m") or screen.get("vq_u")):
                reason = "refusal_with_quote"
            else:
                reason = "fast"
        else:
            reason = "refusal_keyword" if pm == u_idx and cot_guard_enabled and _legacy_over_refusal(cot_unmasked[i]) else "fast"

        screen["route"] = reason
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "fast":
            fast_path_indices.append(i)
        else:
            tribunal_indices.append(i)

    print(f"\nConsistency Gate: {len(fast_path_indices)} ({len(fast_path_indices)/total_samples*100:.1f}%) → Fast Path, {len(tribunal_indices)} ({len(tribunal_indices)/total_samples*100:.1f}%) → Asymmetric Tribunal Path")
    print("Routing reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
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

def _save_prediction(args, record, experiment_signature, pred, vote_trace, verdict):
    if experiment_signature in record and not args.overwrite:
        record[experiment_signature][0].append(pred)
        record[experiment_signature][1].append(vote_trace)
        record[experiment_signature][2].append(verdict)
    else:
        record[experiment_signature] = [[pred], [vote_trace], [verdict]]

TRIBUNAL_DEBATERS = (
    ("Fact-Anchor", FACT_ANCHOR_SYS_PROMPT),
    ("Bias-Auditor", BIAS_AUDITOR_SYS_PROMPT),
)

def _debater_messages(round_prompts, replies, round_idx, agent_id):
    """Chat turns for one debater: its own earlier replies as assistant turns, the others' as labelled user text."""
    names = [name for name, _ in TRIBUNAL_DEBATERS]
    messages = []
    others = ""
    for k in range(round_idx + 1):
        content = f"{others}\n\n{round_prompts[k]}" if others else round_prompts[k]
        messages.append({"role": "user", "content": content})
        if k < round_idx:
            messages.append({"role": "assistant", "content": replies[k][agent_id]})
            others = "Arguments from the other panel members in the previous round:\n\n" + "\n\n".join(
                f"{names[b]}:\n{replies[k][b]}" for b in range(len(names)) if b != agent_id)
    return messages

def _identity_blind_exhibit(item, screen):
    """What the masked and swapped passes found, for the judge; empty when nothing was recorded."""
    lines = []
    pm = screen.get("p_m", -1)
    if pm in (0, 1, 2):
        lines.append(f"An identity-blind reader saw the same context with the people's identities replaced by "
                     f"placeholders and chose option {pm} ({item[f'ans{pm}']}).")
        if screen.get("vq_m"):
            lines.append(f"It cited this sentence of the context: \"{screen['quote_m']}\"")
    if "p_s" in screen and screen.get("p_u", -1) in (0, 1, 2):
        moved = screen["p_s"] == screen["p_s_expected"]
        lines.append("When the two people's identities were swapped in the context, the answer "
                     + ("moved with the person, as a text-grounded answer should." if moved
                        else "did not move with the person."))
    return ("Evidence from counterfactual checks:\n" + "\n".join(lines)) if lines else ""

def execute_asymmetric_tribunal(args, models, tribunal_indices, sample_records, experiment_signature):
    """Component 4: Asymmetric Persona Tribunal (System 2).

    Fact-Anchor and Bias-Auditor debate for num_rounds + 1 rounds, each seeing its own earlier replies as its
    own turns and the other's replies as labelled arguments; the Judicial Arbiter does not vote, and rules once
    on the labelled transcript plus the masked/swapped evidence. --legacy_tribunal restores the old version.
    """
    if getattr(args, "legacy_tribunal", False):
        return _execute_legacy_tribunal(args, models, tribunal_indices, sample_records, experiment_signature)
    if not tribunal_indices:
        return

    num_samples = len(tribunal_indices)
    num_agents = len(TRIBUNAL_DEBATERS)
    print(f"\n--- Asymmetric Persona Tribunal ({num_samples} samples) ---")

    items = [sample_records[i] for i in tribunal_indices]
    round_prompts = [
        [build_prompt(args, item, is_init=(r == 0), logical=(r == 0), masked=False) for r in range(args.num_rounds + 1)]
        for item in items
    ]
    replies = [[[] for _ in range(args.num_rounds + 1)] for _ in range(num_samples)]
    vote_trace = [[[] for _ in range(args.num_rounds + 1)] for _ in range(num_samples)]

    for round_idx in range(args.num_rounds + 1):
        # Agents in a round only see earlier rounds, so the whole round is one batch
        prompts = [
            format_messages(args, _debater_messages(round_prompts[j], replies[j], round_idx, a), sys_prompt)
            for a, (_, sys_prompt) in enumerate(TRIBUNAL_DEBATERS)
            for j in range(num_samples)
        ]
        responses = decode_batch(models, args, prompts, phase=f"tribunal_r{round_idx}")
        for a in range(num_agents):
            for j in range(num_samples):
                resp = responses[a * num_samples + j]
                replies[j][round_idx].append(resp)
                vote_trace[j][round_idx].append([get_pred(resp)])

    names = [name for name, _ in TRIBUNAL_DEBATERS]
    judge_prompts = []
    for j, item in enumerate(items):
        transcript = "\n\n".join(
            f"Round {r + 1}\n" + "\n\n".join(f"{names[a]}:\n{replies[j][r][a]}" for a in range(num_agents))
            for r in range(args.num_rounds + 1)
        )
        parts = [task_prompt(args, item), "Panel debate:\n\n" + transcript]
        exhibit = _identity_blind_exhibit(item, item.get("_screen", {}))
        if exhibit:
            parts.append(exhibit)
        parts.append(build_prompt(args, item, is_init=False, is_summary=True, masked=False))
        judge_prompts.append(format_messages(args, [{"role": "user", "content": "\n\n".join(parts)}],
                                             JUDGE_PERSONA_SYS_PROMPT))

    judge_verdicts = decode_batch(models, args, judge_prompts, phase="tribunal_judge_summary")
    for j, i in enumerate(tribunal_indices):
        _save_prediction(args, sample_records[i], experiment_signature,
                         get_pred(judge_verdicts[j]), vote_trace[j], judge_verdicts[j])

def _execute_legacy_tribunal(args, models, tribunal_indices, sample_records, experiment_signature):
    """Original tribunal: the judge also debates, and all replies are shown unlabelled as one assistant turn."""
    if not tribunal_indices:
        return

    num_samples = len(tribunal_indices)
    print(f"\n--- Asymmetric Persona Tribunal, legacy ({num_samples} samples) ---")

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
    for j in range(num_samples):
        i = tribunal_indices[j]
        item = sample_records[i]
        prompt = build_prompt(args, item, is_init=False, is_summary=True, masked=False)
        judge_synthesis_prompts.append(format_prompt(
            args, prompt,
            sys_prompt=JUDGE_PERSONA_SYS_PROMPT,
            thinking_history=tribunal_transcript[j],
            id=args.summary_id,
        ))

    judge_verdicts = decode_batch(models, args, judge_synthesis_prompts, phase="tribunal_judge_summary")

    for j in range(num_samples):
        i = tribunal_indices[j]
        _save_prediction(args, sample_records[i], experiment_signature,
                         get_pred(judge_verdicts[j]), tribunal_vote_trace[j], judge_verdicts[j])
