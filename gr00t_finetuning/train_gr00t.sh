#!/bin/bash
# ============================================================
# GR00T N1.6 Fine-tuning Script for AgileX Piper Robot
# ============================================================
# 
# This script fine-tunes the GR00T N1.6-3B model on your collected
# pick-and-place demonstrations.
#
# Prerequisites:
# 1. Run setup first: ./setup_gr00t.sh
# 2. Ensure you have a GPU with at least 24GB VRAM (RTX 4090, A6000, etc.)
#
# Usage:
#   ./train_gr00t.sh
# ============================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_DIR="${SCRIPT_DIR}/Isaac-GR00T"
DATASET_PATH="${SCRIPT_DIR}/../datasets/pick_the_white_cup_and_place_it_on_the_red_cup"
MODALITY_CONFIG="${SCRIPT_DIR}/piper_modality.json"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints/piper_finetune"
BASE_MODEL="nvidia/GR00T-N1.6-3B"

# Training hyperparameters
NUM_GPUS=1
MAX_STEPS=2000
SAVE_STEPS=500
BATCH_SIZE=8  # Reduce if OOM, increase if more VRAM

echo "============================================================"
echo "GR00T N1.6 Fine-tuning for AgileX Piper Robot"
echo "============================================================"
echo ""
echo "Dataset:      ${DATASET_PATH}"
echo "Base Model:   ${BASE_MODEL}"
echo "Output:       ${OUTPUT_DIR}"
echo "Max Steps:    ${MAX_STEPS}"
echo "Batch Size:   ${BATCH_SIZE}"
echo ""

# Check if GR00T directory exists
if [ ! -d "${GROOT_DIR}" ]; then
    echo "ERROR: Isaac-GR00T not found at ${GROOT_DIR}"
    echo "Please run: git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T Isaac-GR00T"
    exit 1
fi

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

cd "${GROOT_DIR}"

echo "Starting fine-tuning..."
echo ""

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=${NUM_GPUS}

# Run fine-tuning
uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}" \
    --num-gpus ${NUM_GPUS} \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 3 \
    --save-steps ${SAVE_STEPS} \
    --max-steps ${MAX_STEPS} \
    --global-batch-size ${BATCH_SIZE} \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4

echo ""
echo "============================================================"
echo "Fine-tuning complete!"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate, run:"
echo "  ./evaluate_gr00t.sh"
echo "============================================================"
