#!/usr/bin/env python3
"""
Backward-compatible wrapper for Adaptive-MPT inference.
Enables legacy commands (e.g. `python mpt_inference.py ...`) to execute seamlessly.
"""

from adaptive_mpt import (
    LOGICAL_PROMPT_MASKED,
    LOGICAL_PROMPT_ASYMMETRIC_UNMASKED,
    FACT_ANCHOR_SYS_PROMPT,
    BIAS_AUDITOR_SYS_PROMPT,
    JUDGE_PERSONA_SYS_PROMPT,
    set_tasks_BBQ,
    set_tasks_stereoset,
    mask_entities,
    get_undetermined_index,
    is_classified_as_ambiguous,
    load_dataset_records,
    save_dataset_records,
    LLMDecoder,
    format_prompt,
    decode_batch,
    get_pred,
    build_prompt,
    run_dual_pass_screening,
    route_samples,
    execute_fast_path,
    execute_asymmetric_tribunal,
)
from run_inference import fix_seed, parse_arguments, main

# Backward compatibility alias
Decoder = LLMDecoder

if __name__ == "__main__":
    main()
