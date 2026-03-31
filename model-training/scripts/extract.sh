#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Extract anomaly maps and prediction masks from a trained model checkpoint.
#
# Usage:
#   bash scripts/extract.sh <model> [experiment_name] [gpu_id] [split] [env_name]
#
# Arguments:
#   model            Model name: draem, dinomaly, stfpm, rd4ad, cfa, patchcore,
#                    fastflow, cflow
#                    (deepsvdd and cutpaste are NOT supported — image-level only)
#   experiment_name  (optional) Experiment name matching the training run
#                    (default: <model>_synth23_pelvis_v7_png)
#   gpu_id           (optional) GPU index, or "cpu" (default: 0)
#   split            (optional) Dataset split: test | valid (default: test)
#   env_name         (optional) Conda env name (default: ood)
#
# Examples:
#   bash scripts/extract.sh cfa
#   bash scripts/extract.sh rd4ad rd4ad_synth23_pelvis_v7_png 1
#   bash scripts/extract.sh fastflow my_fastflow_exp 0 valid
#   bash scripts/extract.sh patchcore patchcore_exp cpu test myenv
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AVAILABLE_MODELS="draem dinomaly stfpm rd4ad cfa patchcore fastflow cflow"
UNSUPPORTED_MODELS="deepsvdd cutpaste"

# ── Parse arguments ──────────────────────────────────────────────────────────
MODEL="${1:-patchcore}"
DATASET_NAME="synth23_pelvis_v7_png"
EXP_NAME="${2:-${MODEL}_${DATASET_NAME}}"
GPU_ID="${3:-0}"
SPLIT="${4:-test}"
ENV_NAME="${5:-ood}"

# ── Activate conda env ──────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "[OK] Activated conda env '${ENV_NAME}'"

# ── Validate model ──────────────────────────────────────────────────────────
if echo "${UNSUPPORTED_MODELS}" | grep -qw "${MODEL}"; then
    echo "ERROR: '${MODEL}' does not support extraction (image-level model, no pixel maps)."
    exit 1
fi

if ! echo "${AVAILABLE_MODELS}" | grep -qw "${MODEL}"; then
    echo "ERROR: Unknown model '${MODEL}'."
    echo "Available models: ${AVAILABLE_MODELS}"
    exit 1
fi

# ── Validate config ─────────────────────────────────────────────────────────
CONFIG="${PROJECT_DIR}/config/${MODEL}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file '${CONFIG}' not found."
    exit 1
fi

# ── Locate checkpoint ───────────────────────────────────────────────────────
RESULTS_DIR="${PROJECT_DIR}/results/${MODEL}/${EXP_NAME}"
CKPT="${RESULTS_DIR}/checkpoints/last.ckpt"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: Checkpoint not found at ${CKPT}"
    echo ""
    echo "Available experiments for '${MODEL}':"
    ls -d "${PROJECT_DIR}/results/${MODEL}/"*/ 2>/dev/null | xargs -I{} basename {} || echo "  (none)"
    exit 1
fi

# ── Build output dir ────────────────────────────────────────────────────────
OUTPUT_DIR="${RESULTS_DIR}/extract_${SPLIT}"

# ── Build command ────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

EXTRACT_CMD="python extract.py --config ${CONFIG} --checkpoint ${CKPT} --output_dir ${OUTPUT_DIR} --split ${SPLIT}"

if [[ "${GPU_ID}" == "cpu" ]]; then
    EXTRACT_CMD="${EXTRACT_CMD} --gpu -1"
    echo "[INFO] Running on CPU"
else
    EXTRACT_CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${EXTRACT_CMD}"
    echo "[INFO] Running on GPU: ${GPU_ID}"
fi

echo "[INFO] Model:      ${MODEL}"
echo "[INFO] Experiment: ${EXP_NAME}"
echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Split:      ${SPLIT}"
echo "[INFO] Output:     ${OUTPUT_DIR}"
echo ""
echo "[RUN] ${EXTRACT_CMD}"
echo ""

eval "${EXTRACT_CMD}" || {
    echo ""
    echo "[ERROR] Extraction failed."
    echo "[HINT] If CUDA OOM, try a different GPU or CPU:"
    echo "       bash scripts/extract.sh ${MODEL} ${EXP_NAME} <gpu_id>"
    echo "       bash scripts/extract.sh ${MODEL} ${EXP_NAME} cpu"
    echo ""
    echo "       Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    exit 1
}

echo ""
echo "[DONE] Anomaly maps:      ${OUTPUT_DIR}/anomaly_maps/"
echo "[DONE] Prediction masks:  ${OUTPUT_DIR}/prediction_masks/"
