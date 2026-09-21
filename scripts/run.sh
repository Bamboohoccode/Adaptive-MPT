#!/usr/bin/env bash
set -e

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "Targeting Root Directory: $ROOT_DIR"
cd "$ROOT_DIR"

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET="BBQ"

# Auto-detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=1
fi
[ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1

echo "=========================================================================="
echo ">>> RUNNING ADAPTIVE-MPT BENCHMARK"
echo ">>> Model   : $MODEL"
echo ">>> Dataset : $DATASET (11 categories)"
echo ">>> GPUs    : $NUM_GPUS"
echo "=========================================================================="

python mpt_inference.py \
    --method mpt \
    --models "$MODEL" \
    --dataset "$DATASET" \
    --truncate \
    --random_seed 42 \
    --num_gpus "$NUM_GPUS" \
    --chat_templates "${@:2}"

echo ">>> Evaluating Predictions..."
python mpt_evaluate.py \
    --method mpt \
    --models "$MODEL" \
    --dataset "$DATASET" \
    --truncate
