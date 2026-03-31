#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Train a specified model on a specified dataset.
#
# Usage:
#   bash scripts/train.sh <model> <dataset> [experiment_name] [gpu_ids]
#
# Arguments:
#   model            Model name: draem, dinomaly, deepsvdd, cutpaste, stfpm,
#                    rd4ad, cfa, patchcore, fastflow, cflow
#   dataset          Path to dataset folder (e.g. ./datasets/synth23_pelvis_v7_png)
#   experiment_name  (optional) Experiment name for output folder (default: <model>_<dataset_basename>)
#   gpu_ids          (optional) GPU ids, comma-separated (default: 0)
#                    Use "cpu" to run on CPU
#   env_name         (optional) Conda env name (default: ood)
#
# Examples:
#   bash scripts/train.sh rd4ad ./datasets/synth23_pelvis_v7_png
#   bash scripts/train.sh fastflow ./datasets/synth23_pelvis_v7_png exp1 0,1,2
#   bash scripts/train.sh patchcore ./datasets/synth23_pelvis_v7_png exp1 cpu
#   bash scripts/train.sh draem ./datasets/synth23_pelvis_v7_png my_draem 1 myenv
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${5:-ood}"

# ── Activate conda env ──────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "[OK] Activated conda env '${ENV_NAME}'"

AVAILABLE_MODELS="draem dinomaly deepsvdd cutpaste stfpm rd4ad cfa patchcore fastflow cflow"

# ── Parse arguments ──────────────────────────────────────────────────────────
MODEL="${1:-patchcore}"
DATASET="${2:-${PROJECT_DIR}/datasets/synth23_pelvis_v7_png}"
DATASET_NAME="$(basename "${DATASET}")"
EXP_NAME="${3:-${MODEL}_${DATASET_NAME}}"
GPU_IDS="${4:-0}"

# ── Validate model ──────────────────────────────────────────────────────────
if ! echo "${AVAILABLE_MODELS}" | grep -qw "${MODEL}"; then
    echo "ERROR: Unknown model '${MODEL}'."
    echo "Available models: ${AVAILABLE_MODELS}"
    exit 1
fi

# ── Validate dataset ────────────────────────────────────────────────────────
if [[ ! -d "${DATASET}" ]]; then
    echo "ERROR: Dataset path '${DATASET}' does not exist."
    exit 1
fi

# Check dataset structure
ERRORS=()
if [[ ! -d "${DATASET}/train/good" ]]; then
    ERRORS+=("  missing: train/good/  (normal training images)")
fi
if [[ ! -d "${DATASET}/valid/good" ]]; then
    ERRORS+=("  missing: valid/good/  (normal validation images)")
fi
if [[ ! -d "${DATASET}/valid/Ungood" ]]; then
    ERRORS+=("  missing: valid/Ungood/  (abnormal validation images)")
fi

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo "ERROR: Dataset '${DATASET}' has incorrect structure:"
    for e in "${ERRORS[@]}"; do echo "$e"; done
    echo ""
    echo "Expected layout:"
    echo "  ${DATASET}/"
    echo "  ├── train/good/              # normal training images"
    echo "  └── valid/"
    echo "      ├── good/img/            # normal validation images"
    echo "      └── Ungood/"
    echo "          ├── img/             # abnormal validation images"
    echo "          └── label/           # binary masks (optional)"
    exit 1
fi

# Check train/good has images
TRAIN_COUNT=$(find "${DATASET}/train/good" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.nii.gz" -o -name "*.nii" \) -print -quit | wc -l)
if [[ "${TRAIN_COUNT}" -eq 0 ]]; then
    echo "[WARN] No images found in ${DATASET}/train/good/"
fi

# ── Validate config ─────────────────────────────────────────────────────────
CONFIG="${PROJECT_DIR}/config/${MODEL}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file '${CONFIG}' not found."
    exit 1
fi

# ── Build command ────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

TRAIN_CMD="python train.py --config ${CONFIG} --data_root ${DATASET} --name ${EXP_NAME}"

# Handle accelerator
if [[ "${GPU_IDS}" == "cpu" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --accelerator cpu"
    echo "[INFO] Running on CPU"
else
    TRAIN_CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} ${TRAIN_CMD}"
    echo "[INFO] Running on GPU: ${GPU_IDS}"
fi

echo "[INFO] Model:      ${MODEL}"
echo "[INFO] Dataset:    ${DATASET}"
echo "[INFO] Experiment: ${EXP_NAME}"
echo "[INFO] Config:     ${CONFIG}"
echo ""
echo "[RUN] ${TRAIN_CMD}"
echo ""

OUTPUT_DIR="${PROJECT_DIR}/results/${MODEL}/${EXP_NAME}"

eval "${TRAIN_CMD}" || {
    echo ""
    echo "[ERROR] Training failed."
    echo "[HINT] If CUDA OOM, try a different GPU:"
    echo "       bash scripts/train.sh ${MODEL} ${DATASET} ${EXP_NAME} <gpu_id>"
    echo ""
    echo "       Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    exit 1
}

echo ""
echo "[DONE] Results saved to: ${OUTPUT_DIR}"
