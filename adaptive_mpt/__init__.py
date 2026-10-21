"""Adaptive-MPT: Efficient & Debiaised Reasoning via Dual-Pass Screening and Asymmetric Persona Tribunal."""

from adaptive_mpt.prompts import (
    LOGICAL_PROMPT_MASKED,
    LOGICAL_PROMPT_ASYMMETRIC_UNMASKED,
    FACT_ANCHOR_SYS_PROMPT,
    BIAS_AUDITOR_SYS_PROMPT,
    JUDGE_PERSONA_SYS_PROMPT,
)
from adaptive_mpt.data import (
    set_tasks_BBQ,
    set_tasks_stereoset,
    set_tasks_BiasFreeBench,
    mask_entities,
    swap_entities,
    get_undetermined_index,
    is_classified_as_ambiguous,
    load_dataset_records,
    save_dataset_records,
)
from adaptive_mpt.models import (
    LLMDecoder,
    format_prompt,
    decode_batch,
    get_pred,
)
from adaptive_mpt.pipeline import (
    build_prompt,
    run_dual_pass_screening,
    route_samples,
    execute_fast_path,
    execute_asymmetric_tribunal,
)

__version__ = "2.0.0"
__all__ = [
    "LOGICAL_PROMPT_MASKED",
    "LOGICAL_PROMPT_ASYMMETRIC_UNMASKED",
    "FACT_ANCHOR_SYS_PROMPT",
    "BIAS_AUDITOR_SYS_PROMPT",
    "JUDGE_PERSONA_SYS_PROMPT",
    "set_tasks_BBQ",
    "set_tasks_stereoset",
    "mask_entities",
    "swap_entities",
    "get_undetermined_index",
    "is_classified_as_ambiguous",
    "load_dataset_records",
    "save_dataset_records",
    "LLMDecoder",
    "format_prompt",
    "decode_batch",
    "get_pred",
    "build_prompt",
    "run_dual_pass_screening",
    "route_samples",
    "execute_fast_path",
    "execute_asymmetric_tribunal",
]
