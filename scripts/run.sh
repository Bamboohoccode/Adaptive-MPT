SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "Targeting Root Directory: $ROOT_DIR"
cd "$ROOT_DIR"

MODE="${1:-dual_pass}"   # "dual_pass" or "pipeline"
MODEL="${2:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET="BBQ"

# Auto-detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=1
fi
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1

echo "=========================================================================="
echo ">>> RUNNING BBQ BENCHMARK (Mode: $MODE)"
echo ">>> Model   : $MODEL"
echo ">>> Dataset : $DATASET (11 categories)"
echo ">>> GPUs    : $NUM_GPUS"
echo "=========================================================================="

if [ "$MODE" = "dual_pass" ]; then
    echo ">>> Executing Dual-Pass Counterfactual Batch Inference & Fast-Path Reprompting..."
    python check_dual_pass.py \
        --models "$MODEL" \
        --dataset "$DATASET" \
        --num_gpus "$NUM_GPUS" \
        --gpu_memory_utilization 0.85 \
        --max_model_len 4096 \
        --asymmetric \
        --chat_templates \
        --output_json ./predictions/BBQ_dual_pass_results.json "${@:3}"

elif [ "$MODE" = "pipeline" ]; then
    echo ">>> Executing Adaptive-MPT End-to-End Pipeline with 3-Persona Tribunal..."
    python mpt_inference.py \
        --method mpt \
        --models "$MODEL" \
        --dataset "$DATASET" \
        --router_method context_sufficiency \
        --fast_path_method logical_reprompting \
        --debate_mode asymmetric_tribunal \
        --num_rounds 3 \
        --num_gpus "$NUM_GPUS" \
        --gpu_memory_utilization 0.85 \
        --max_model_len 4096 \
        --chat_templates "${@:3}"

    echo ">>> Evaluating Predictions..."
    python mpt_evaluate.py \
        --method mpt \
        --models "$MODEL" \
        --dataset "$DATASET" \
        --chat_templates
else
    echo "Unknown MODE: $MODE. Use 'dual_pass' or 'pipeline'."
    exit 1
fi