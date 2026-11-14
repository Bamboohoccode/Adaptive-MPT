import json
import re
from pathlib import Path
from adaptive_mpt.prompts import (
    DEFAULT_SYS_PROMPT,
    QUESTION_PROMPT_BBQ,
    QUESTION_PROMPT_STEREOSET,
    CONSTRAINT_PROMPT_BBQ,
    CONSTRAINT_PROMPT_STS,
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
    FEW_SHOT_UNMASKED_DISAMBIG,
    FEW_SHOT_UNMASKED_AMBIG,
    FEW_SHOT_MASKED_DISAMBIG,
    FEW_SHOT_MASKED_AMBIG,
    FACT_ANCHOR_SYS_PROMPT,
    BIAS_AUDITOR_SYS_PROMPT,
    JUDGE_PERSONA_SYS_PROMPT,
    SUMMARY_PROMPT,
    REVIEW_PROMPT
)
from adaptive_mpt.data import mask_entities, get_undetermined_index
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

FEW_SHOT_DATASETS = ("BBQ", "BiasFreeBench", "biasfreebench")

def _permute_options(item, new_pos):
    """Copy of a BBQ-style item with options moved: new_pos maps old option index -> new index."""
    out = dict(item)
    order = sorted(new_pos, key=new_pos.get)
    for old in order:
        out[f"ans{new_pos[old]}"] = item[f"ans{old}"]
    out["answer_info"] = {f"ans{new_pos[old]}": item["answer_info"][f"ans{old}"] for old in order}
    out["label"] = new_pos[item["label"]]
    return out

def _deciding_sentence(item, ambig_context, entity_map, masked):
    """First sentence of the disambiguating text that valid_quote accepts, or None."""
    context = item["context"]
    addition = context[len(ambig_context):] if context.startswith(ambig_context) else context
    for sent in re.split(r"(?<=[.!?])\s+", addition.strip()):
        if valid_quote(f"<quote>{sent}</quote>", item, entity_map, masked=masked):
            return sent
    return None

def _demo_response(item, quote, masked):
    """Worked answer for one few-shot example, from the FEW_SHOT_* templates in prompts.py."""
    label = item["label"]
    u = get_undetermined_index(item)
    people = [re.sub(r"^The ", "the ", str(item[f"ans{k}"]).strip()) for k in range(3) if k != u]
    if masked:
        template = FEW_SHOT_MASKED_DISAMBIG if quote else FEW_SHOT_MASKED_AMBIG
    else:
        template = FEW_SHOT_UNMASKED_DISAMBIG if quote else FEW_SHOT_UNMASKED_AMBIG
    return template.format(question=str(item["question"]).strip(), people=" and ".join(people), quote=quote,
                           label=label, option=item[f"ans{label}"])

def build_few_shot_demos(args):
    """--few_shot K worked examples for the screening passes, from --few_shot_dir (default data/synthetic_bbq).

    Demo i is disambiguated when i is even and ambiguous when odd, alternates neg/nonneg questions, cycles through
    categories, and has its options permuted so the correct answer sits at position i % 3 (the synthetic files always
    put "unknown" last). Each demo exists in an unmasked form (used by the unmasked pass) and a masked form (used by the
    masked pass).
    Returns {"unmasked": [(item, response)], "masked": [(item, response)]}, or None when few-shot is off.
    """
    k = getattr(args, "few_shot", 0)
    if k <= 0 or args.dataset not in FEW_SHOT_DATASETS:
        return None
    cached = getattr(args, "_few_shot_demos", None)
    if cached is not None:
        return cached

    files = sorted(Path(getattr(args, "few_shot_dir", "data/synthetic_bbq")).glob("*.jsonl"))
    groups = []
    for path in files:
        rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        groups.append([rows[j:j + 4] for j in range(0, len(rows) - 3, 4)])
    demos, used = {"unmasked": [], "masked": []}, set()
    for i in range(k):
        disambig, polarity, target = i % 2 == 0, ("neg" if (i // 2) % 2 == 0 else "nonneg"), i % 3
        found = False
        for f in range(len(files)):
            fi = (i + f) % len(files)
            for gi, group in enumerate(groups[fi]):
                pick = [r for r in group if r["question_polarity"] == polarity]
                row = next((r for r in pick if r["context_condition"] == ("disambig" if disambig else "ambig")), None)
                ambig = next((r for r in pick if r["context_condition"] == "ambig"), None)
                if row is None or ambig is None or (fi, gi, polarity, disambig) in used:
                    continue
                others = [o for o in range(3) if o != row["label"]]
                new_pos = dict(zip([row["label"]] + others, [target] + [n for n in range(3) if n != target]))
                item, ambig_item = _permute_options(row, new_pos), _permute_options(ambig, new_pos)
                masked_item, masked_ambig = mask_entities(item), mask_entities(ambig_item)
                entity_map = masked_item["_entity_map"]
                quote_u = quote_m = None
                if disambig:
                    quote_u = _deciding_sentence(item, ambig_item["context"], entity_map, masked=False)
                    quote_m = _deciding_sentence(masked_item, masked_ambig["context"], entity_map, masked=True)
                    if not (quote_u and quote_m):
                        continue
                demos["unmasked"].append((item, _demo_response(item, quote_u, masked=False)))
                demos["masked"].append((masked_item, _demo_response(masked_item, quote_m, masked=True)))
                used.add((fi, gi, polarity, disambig))
                found = True
                break
            if found:
                break
        if not found:
            raise ValueError(f"--few_shot {k}: no usable example left for demo {i + 1} in {files[0].parent if files else 'few_shot_dir'}")
    print(f"Few-shot: {k} demos, correct options at " + ", ".join(str(it["label"]) for it, _ in demos["unmasked"]))
    args._few_shot_demos = demos
    return demos

def _screen_prompt(args, item, masked, demos):
    """Screening prompt; with demos, each demo is a user turn plus a worked assistant turn before the real question."""
    prompt = build_prompt(args, item, is_init=True, logical=True, masked=masked)
    if not demos:
        return format_prompt(args, prompt, sys_prompt=DEFAULT_SYS_PROMPT)
    messages = []
    for demo_item, demo_response in demos["masked" if masked else "unmasked"]:
        messages.append({"role": "user", "content": build_prompt(args, demo_item, is_init=True, logical=True, masked=masked)})
        messages.append({"role": "assistant", "content": demo_response})
    messages.append({"role": "user", "content": prompt})
    return format_messages(args, messages, DEFAULT_SYS_PROMPT)

def run_dual_pass_screening(args, models, sample_records):
    """Component 1: Counterfactual Dual-Pass Batch Inference (masked and unmasked passes in one 2N batch).

    Pass 1 (Masked Logic): [Person A], [Person B] with LOGICAL_PROMPT_MASKED -> P_M
    Pass 2 (Unmasked CoT): Original identities with LOGICAL_PROMPT_ASYMMETRIC_UNMASKED -> P_U, R_U
    With --few_shot K, both passes get the same K worked examples (masked ones for the masked pass).
    Per-sample screening details (predictions, quote checks, raw masked response) go to record["_screen"].
    """
    total_samples = len(sample_records)
    print("\n--- Phase 1: Counterfactual Dual-Pass Screening (masked + unmasked) ---")

    counterfactual_masked_inputs = [mask_entities(it) for it in sample_records]
    demos = build_few_shot_demos(args)

    masked_prompts = []
    unmasked_prompts = []
    for i in range(total_samples):
        masked_prompts.append(_screen_prompt(args, counterfactual_masked_inputs[i], True, demos))
        unmasked_prompts.append(_screen_prompt(args, sample_records[i], False, demos))

    # Single unified batch execution on vLLM (2N prompts)
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

        entity_map = counterfactual_masked_inputs[i]["_entity_map"]
        screen = {
            "p_m": pm,
            "p_u": pu,
            "vq_m": valid_quote(masked_responses[i], counterfactual_masked_inputs[i], entity_map, masked=True),
            "vq_u": valid_quote(unmasked_responses[i], sample_records[i], entity_map),
            "quote_m": extract_quote(masked_responses[i]),
            "masked_response": masked_responses[i],
        }
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

    Tribunal when: the masked and unmasked answers differ; either is unparseable; both pick a person but the unmasked pass quoted no real deciding sentence (a shared guess);
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
    """What the masked pass found, for the judge; empty when nothing was recorded."""
    lines = []
    pm = screen.get("p_m", -1)
    if pm in (0, 1, 2):
        lines.append(f"An identity-blind reader saw the same context with the people's identities replaced by "
                     f"placeholders and chose option {pm} ({item[f'ans{pm}']}).")
        if screen.get("vq_m"):
            lines.append(f"It cited this sentence of the context: \"{screen['quote_m']}\"")
    return ("Evidence from counterfactual checks:\n" + "\n".join(lines)) if lines else ""

def execute_asymmetric_tribunal(args, models, tribunal_indices, sample_records, experiment_signature):
    """Component 4: Asymmetric Persona Tribunal (System 2).

    Fact-Anchor and Bias-Auditor debate for num_rounds + 1 rounds, each seeing its own earlier replies as its
    own turns and the other's replies as labelled arguments; the Judicial Arbiter does not vote, and rules once
    on the labelled transcript plus the masked-pass evidence. --legacy_tribunal restores the old version.
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

    active = list(range(num_samples))
    for round_idx in range(args.num_rounds + 1):
        if not active:
            break
        # Agents in a round only see earlier rounds, so the whole round is one batch
        prompts = [
            format_messages(args, _debater_messages(round_prompts[j], replies[j], round_idx, a), sys_prompt)
            for a, (_, sys_prompt) in enumerate(TRIBUNAL_DEBATERS)
            for j in active
        ]
        responses = decode_batch(models, args, prompts, phase=f"tribunal_r{round_idx}")
        for a in range(num_agents):
            for k, j in enumerate(active):
                resp = responses[a * len(active) + k]
                replies[j][round_idx].append(resp)
                vote_trace[j][round_idx].append([get_pred(resp)])
        # Fast convergence: both debaters give the same parseable answer, so further rounds add nothing (judge still rules)
        converged = {j for j in active if (p := [v[0] for v in vote_trace[j][round_idx]])[0] != -1 and len(set(p)) == 1}
        active = [j for j in active if j not in converged]
    for j in range(num_samples):
        replies[j] = [r for r in replies[j] if r]
        vote_trace[j] = [v for v in vote_trace[j] if v]

    names = [name for name, _ in TRIBUNAL_DEBATERS]
    judge_prompts = []
    for j, item in enumerate(items):
        transcript = "\n\n".join(
            f"Round {r + 1}\n" + "\n\n".join(f"{names[a]}:\n{replies[j][r][a]}" for a in range(num_agents))
            for r in range(len(replies[j]))
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
    active = list(range(num_samples))
    for round_idx in range(args.num_rounds + 1):
        if not active:
            break
        round_responses = {j: [] for j in active}

        for agent_id in range(max_agents):
            formatted_prompts = []
            for j in active:
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

            responses = decode_batch(models, args, formatted_prompts, phase=f"tribunal_r{round_idx}_a{agent_id}")
            for k, j in enumerate(active):
                resp = responses[k]
                tribunal_vote_trace[j][round_idx].append([get_pred(resp)])
                round_responses[j].append(resp)

        for j in active:
            tribunal_transcript[j][round_idx].extend(round_responses[j])
        # Fast convergence: all agents give the same parseable answer, so further rounds add nothing (judge still rules)
        active = [j for j in active
                  if (p := [v[0] for v in tribunal_vote_trace[j][round_idx]])[0] == -1 or len(set(p)) > 1]
    for j in range(num_samples):
        tribunal_transcript[j] = [r for r in tribunal_transcript[j] if r]
        tribunal_vote_trace[j] = [v for v in tribunal_vote_trace[j] if v]

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
