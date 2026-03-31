#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline: setup → train → extract
#
# Usage:
#   bash scripts/run_pipeline.sh [model] [dataset] [experiment_name] [gpu_ids] [env_name]
#
# All arguments are optional and share the same defaults as train.sh / extract.sh.
#
# Examples:
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh cfa
#   bash scripts/run_pipeline.sh rd4ad ./datasets/synth23_pelvis_v7_png my_exp 1
#   bash scripts/run_pipeline.sh fastflow ./datasets/synth23_pelvis_v7_png exp1 0,1,2 myenv
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-patchcore}"
DATASET="${2:-${PROJECT_DIR}/datasets/synth23_pelvis_v7_png}"
DATASET_NAME="$(basename "${DATASET}")"
EXP_NAME="${3:-${MODEL}_${DATASET_NAME}}"
GPU_IDS="${4:-0}"
ENV_NAME="${5:-ood}"

# extract only uses first GPU
GPU_FIRST="${GPU_IDS%%,*}"

echo "============================================="
echo "  MROOD-TRAIN Pipeline"
echo "  Model:      ${MODEL}"
echo "  GPU:        ${GPU_IDS}"
echo "  Env:        ${ENV_NAME}"
echo "============================================="
echo ""

# ── 1. Setup ─────────────────────────────────────────────────────────────────
echo ">>> Step 1/3: Setup environment"
echo ""
bash "${SCRIPT_DIR}/setup.sh" "${ENV_NAME}"
echo ""

# ── 2. Train ─────────────────────────────────────────────────────────────────
echo ">>> Step 2/3: Train"
echo ""

bash "${SCRIPT_DIR}/train.sh" "${MODEL}" "${DATASET}" "${EXP_NAME}" "${GPU_IDS}" "${ENV_NAME}"
echo ""

# ── 3. Extract ───────────────────────────────────────────────────────────────
# deepsvdd and cutpaste don't support extraction
UNSUPPORTED="deepsvdd cutpaste"
if echo "${UNSUPPORTED}" | grep -qw "${MODEL}"; then
    echo ">>> Step 3/3: Extract (skipped — ${MODEL} is image-level only)"
else
    echo ">>> Step 3/3: Extract"
    echo ""

    bash "${SCRIPT_DIR}/extract.sh" "${MODEL}" "${EXP_NAME}" "${GPU_FIRST}" "test" "${ENV_NAME}"
fi

echo ""
echo "============================================="
echo "  Pipeline complete!"
echo "============================================="
