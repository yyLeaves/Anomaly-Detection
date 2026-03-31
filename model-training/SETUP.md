# Environment Setup & Test Run Commands

## Installation

```bash
conda create -n ood python=3.10 -y

# Install GPU torch
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install other requirements
pip install -r requirements-all.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Patch anomalib FastFlow to support resnet50 backbone (required for Exp 3)
sed -i 's/backbone in {"resnet18", "wide_resnet50_2"}/backbone in {"resnet18", "resnet50", "wide_resnet50_2"}/' \
    $(python -c "import anomalib.models.image.fastflow.torch_model as m; print(m.__file__)")
```

## Dataset

Three input variants, all sharing the same folder layout:

| Dataset | Format | Path |
|---------|--------|------|
| `synth23_pelvis_v7_png` | PNG (replicated channel) | `datasets/synth23_pelvis_v7_png` |
| `synth23_pelvis_v7_nifti_3ch_rep` | NIfTI 3-channel replicated | `datasets/synth23_pelvis_v7_nifti_3ch_rep` |
| `synth23_pelvis_v7_nifti_con` | NIfTI adjacent-slice context | `datasets/synth23_pelvis_v7_nifti_con` |

NIfTI datasets are auto-converted to a PNG cache (`{dataset}_png_cache/`) on first run.

### Switching dataset variant

All commands default to the PNG dataset. To use a different variant, add `--data_root`:

```bash
# replicated-channel NIfTI
python train.py --config config/rd4ad.yaml --name exp_3ch --data_root ./datasets/synth23_pelvis_v7_nifti_3ch_rep

# adjacent-context NIfTI
python train.py --config config/rd4ad.yaml --name exp_con --data_root ./datasets/synth23_pelvis_v7_nifti_con
```

Same flag works for `extract.py`.

---

## Test Runs

```bash
conda activate ood
cd /home/user/lyeyang/projects/mrood-bmad/MROOD-TRAIN

# ── DRAEM ─────────────────────────────────────────────────────────────────────
# NOTE: DTD textures (~600 MB) auto-downloaded to datasets/dtd on first run
CUDA_VISIBLE_DEVICES=1 python train.py --config config/draem.yaml --name test_draem

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/draem.yaml \
    --checkpoint results/draem/test_draem/checkpoints/last.ckpt \
    --output_dir results/draem/test_draem/extract_out

# ── Dinomaly ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python train.py --config config/dinomaly.yaml --name test_dinomaly

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/dinomaly.yaml \
    --checkpoint results/dinomaly/test_dinomaly/checkpoints/last.ckpt \
    --output_dir results/dinomaly/test_dinomaly/extract_out

# ── DeepSVDD ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python train.py --config config/deepsvdd.yaml --name test_deepsvdd

# ── CutPaste ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python train.py --config config/cutpaste.yaml --name test_cutpaste

# ── STFPM ─────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python train.py --config config/stfpm.yaml --name test_stfpm

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/stfpm.yaml \
    --checkpoint results/stfpm/test_stfpm/checkpoints/last.ckpt \
    --output_dir results/stfpm/test_stfpm/extract_out

# ── RD4AD ─────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python train.py --config config/rd4ad.yaml --name test_rd4ad

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/rd4ad.yaml \
    --checkpoint results/rd4ad/test_rd4ad/checkpoints/last.ckpt \
    --output_dir results/rd4ad/test_rd4ad/extract_out

# ── CFA ──────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python train.py --config config/cfa.yaml --name test_cfa

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/cfa.yaml \
    --checkpoint results/cfa/test_cfa/checkpoints/last.ckpt \
    --output_dir results/cfa/test_cfa/extract_out

# ── PatchCore ────────────────────────────────────────────────────────────────
# NOTE: OOM on single RTX 2080 Ti (11 GB); run on CPU instead (requires ~100 GB RAM)
python train.py --config config/patchcore.yaml --accelerator cpu --name test_patchcore

python extract.py --config config/patchcore.yaml \
    --checkpoint results/patchcore/test_patchcore/checkpoints/last.ckpt \
    --output_dir results/patchcore/test_patchcore/extract_out

# ── FastFlow ──────────────────────────────────────────────────────────────────
# Exp 2: ImageNet wide_resnet50_2 (default config)
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --config config/fastflow.yaml --name test_fastflow

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/fastflow.yaml \
    --checkpoint results/fastflow/test_fastflow/checkpoints/last.ckpt \
    --output_dir results/fastflow/test_fastflow/extract_out

# Exp 3: RadImageNet ResNet50 backbone
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --config config/fastflow.yaml --name test_fastflow_radimagenet \
    --backbone radimagenet_resnet50 \
    --radimagenet_ckpt RadImageNet_pytorch/ResNet50.pt

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/fastflow.yaml \
    --checkpoint results/fastflow/test_fastflow_radimagenet/checkpoints/last.ckpt \
    --output_dir results/fastflow/test_fastflow_radimagenet/extract_out \
    --backbone radimagenet_resnet50 \
    --radimagenet_ckpt RadImageNet_pytorch/ResNet50.pt

# ── CFlow ─────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --config config/cflow.yaml --name test_cflow

CUDA_VISIBLE_DEVICES=1 python extract.py --config config/cflow.yaml \
    --checkpoint results/cflow/test_cflow/checkpoints/last.ckpt \
    --output_dir results/cflow/test_cflow/extract_out
```
