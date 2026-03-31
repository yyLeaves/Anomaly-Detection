#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Setup conda environment for MROOD-TRAIN.
#
# What it does:
#   1. Creates the conda env if it doesn't exist
#   2. Installs PyTorch + CUDA
#   3. Installs requirements-all.txt (anomalib, nibabel, etc.)
#
# Usage:
#   bash scripts/setup.sh              # uses default env name 'ood'
#   bash scripts/setup.sh myenv        # uses custom env name 'myenv'
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="${1:-ood}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_INDEX="https://download.pytorch.org/whl/cu124"

if [[ "$ENV_NAME" == "ood" && $# -eq 0 ]]; then
    echo "[INFO] Using default env name 'ood'. To customize: bash scripts/setup.sh <env_name>"
fi

echo "[INFO] CUDA index is set to cu124 (configured for rolf server)."
echo "       If running on a different machine, you may need to change CUDA_INDEX"
echo "       in this script to match your CUDA version."

# ── Locate conda ─────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda/Anaconda first."
    exit 1
fi

eval "$(conda shell.bash hook)"

# ── 1. Create env ─────────────────────────────────────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "[OK] Conda env '${ENV_NAME}' exists."
else
    echo "[SETUP] Creating conda env '${ENV_NAME}' (Python 3.10)..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"
echo "[OK] Activated '${ENV_NAME}' ($(python --version))"

# ── 2. Install PyTorch + CUDA ────────────────────────────────────────────────
echo "[SETUP] Installing torch 2.6.0+cu124 and torchvision..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url "${CUDA_INDEX}"

# ── 3. Install all requirements ──────────────────────────────────────────────
echo "[SETUP] Installing requirements-all.txt..."
pip install -r "${PROJECT_DIR}/requirements-all.txt" \
    --extra-index-url "${CUDA_INDEX}"

# ── 4. Download RadImageNet weights ──────────────────────────────────────────
RADIMAGENET_DIR="${PROJECT_DIR}/RadImageNet_pytorch"
RADIMAGENET_CKPT="${RADIMAGENET_DIR}/ResNet50.pt"
RADIMAGENET_URL="https://huggingface.co/Lab-Rasool/RadImageNet/resolve/main/ResNet50.pt"

if [[ -f "${RADIMAGENET_CKPT}" ]]; then
    echo "[OK] RadImageNet ResNet50 weights found at ${RADIMAGENET_CKPT}"
else
    echo "[SETUP] Downloading RadImageNet ResNet50 weights..."
    mkdir -p "${RADIMAGENET_DIR}"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${RADIMAGENET_CKPT}" "${RADIMAGENET_URL}"
    elif command -v curl &>/dev/null; then
        curl -L -o "${RADIMAGENET_CKPT}" "${RADIMAGENET_URL}"
    else
        echo "[WARN] Neither wget nor curl found. Download manually:"
        echo "       ${RADIMAGENET_URL}"
        echo "       Place at: ${RADIMAGENET_CKPT}"
    fi
fi

# ── 5. Check datasets ────────────────────────────────────────────────────────
DATASETS_DIR="${PROJECT_DIR}/datasets"
if [[ -d "${DATASETS_DIR}" ]] && find "${DATASETS_DIR}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
    echo "[OK] Datasets found in ${DATASETS_DIR}:"
    ls -d "${DATASETS_DIR}"/*/ 2>/dev/null | xargs -I{} basename {}
else
    echo "[WARN] No datasets found in ${DATASETS_DIR}/"
    echo "       Place your dataset folders there before training. Expected layout:"
    echo "         datasets/<dataset_name>/train/good/          # normal training images (PNG or NIfTI)"
    echo "         datasets/<dataset_name>/valid/good/img/      # normal validation images"
    echo "         datasets/<dataset_name>/valid/Ungood/img/    # abnormal validation images"
    echo "         datasets/<dataset_name>/valid/Ungood/label/  # binary masks"
    echo "       NIfTI inputs are auto-converted to PNG on first run."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Environment '${ENV_NAME}' is ready ==="
echo "  Python:   $(python --version 2>&1)"
echo "  torch:    $(python -c 'import torch; print(torch.__version__)')"
echo "  anomalib: $(python -c 'import anomalib; print(anomalib.__version__)')"
echo ""
echo "To use:  conda activate ${ENV_NAME} && cd ${PROJECT_DIR}"
