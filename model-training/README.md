# model-training

Unified training and feature-extraction pipeline for medical anomaly detection models.

## Supported Models

| Model | Family |
|-------|--------|
| `draem` | Reconstruction |
| `dinomaly` | Reconstruction |
| `deepsvdd` | One-Class |
| `cutpaste` | One-Class |
| `stfpm` | Knowledge Distillation |
| `rd4ad` | Knowledge Distillation |
| `cfa` | Memory Bank |
| `patchcore` | Memory Bank |
| `fastflow` | Normalizing Flow |
| `cflow` | Normalizing Flow |

## Dataset Structure

All models expect the dataset root to follow this layout:

```
<data_root>/
├── train/
│   └── good/               # Normal training images (PNG)
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
└── valid/
    ├── good/
    │   └── img/            # Normal validation images
    │       ├── img_001.png
    │       └── ...
    └── Ungood/
        ├── img/            # Abnormal validation images
        │   ├── img_001.png
        │   └── ...
        └── label/          # Binary segmentation masks (same filename as img/)
            ├── img_001.png
            └── ...
```

**Notes:**
- Images should be PNG format (NIfTI inputs are auto-converted via `data/dataset_conversion.py`).
- `label/` masks are optional for models that don't use pixel-level supervision (draem, dinomaly, fastflow, cflow, deepsvdd, cutpaste).
- `train/good/` contains only normal (healthy) images — no anomalous samples used during training.

## Quick Start (Shell Scripts)

### Full Pipeline (setup + train + extract)

```bash
bash scripts/run_pipeline.sh [model] [dataset] [experiment_name] [gpu_ids] [env_name]

# Examples:
bash scripts/run_pipeline.sh                              # patchcore, default dataset, GPU 0
bash scripts/run_pipeline.sh cfa                          # CFA, default dataset, GPU 0
bash scripts/run_pipeline.sh rd4ad ./datasets/my_data exp1 1
bash scripts/run_pipeline.sh fastflow ./datasets/my_data exp1 0,1,2 myenv
```

### Individual Steps
> For manual installation steps and per-model test run commands, see [SETUP.md](SETUP.md).

```bash
# 1. Setup environment (creates conda env, installs deps, downloads RadImageNet weights)
#    For manual setup steps, see SETUP.md
bash scripts/setup.sh [env_name]

# 2. Train a model
bash scripts/train.sh [model] [dataset] [experiment_name] [gpu_ids] [env_name]

# 3. Extract anomaly maps + prediction masks
bash scripts/extract.sh [model] [experiment_name] [gpu_id] [split] [env_name]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model` | `patchcore` | Model name (see table above) |
| `dataset` | `./datasets/synth23_pelvis_v7_png` | Path to dataset folder |
| `experiment_name` | `<model>_<dataset_basename>` | Output subfolder name |
| `gpu_ids` | `0` | GPU index(es), comma-separated. Use `cpu` for CPU |
| `env_name` | `ood` | Conda environment name |
| `split` | `test` | Dataset split for extraction (`test` or `valid`) |

**Note:** `extract.sh` does not support `deepsvdd` or `cutpaste` (image-level models without pixel maps).

## Usage (Python CLI)

### Training

```bash
python train.py --config config/<model>.yaml \
                --data_root /path/to/dataset \
                --name my_experiment
```

Override YAML hyperparameters via CLI flags:

```bash
python train.py --config config/rd4ad.yaml \
                --data_root /data/RESC \
                --name resc_rd4ad \
                --epochs 200 \
                --batch_size 8 \
                --accelerator gpu
```

### Feature Extraction

```bash
python extract.py --config config/<model>.yaml \
                  --checkpoint results/<model>/my_experiment/checkpoints/last.ckpt \
                  --data_root /path/to/dataset \
                  --output_dir /path/to/outputs
```

## RadImageNet Backbone

FastFlow supports using a [RadImageNet](https://github.com/BMEII-AI/RadImageNet)-pretrained ResNet50 as backbone instead of the default ImageNet weights.

`scripts/setup.sh` downloads the weights automatically to `RadImageNet_pytorch/ResNet50.pt`.

To use RadImageNet as backbone, set in the model's config YAML:

```yaml
model:
  backbone: radimagenet_resnet50
  radimagenet_ckpt: ~              # auto-resolved to RadImageNet_pytorch/ResNet50.pt
```

## Directory Structure

```
model-training/
├── README.md                 # This file
├── SETUP.md                  # Step-by-step install commands and per-model test runs
├── train.py                  # Unified training entry point
├── extract.py                # Unified feature extraction entry point
├── scripts/
│   ├── setup.sh              # Environment setup (conda + deps + RadImageNet weights)
│   ├── train.sh              # Train a model on a dataset
│   ├── extract.sh            # Extract anomaly maps + prediction masks
│   └── run_pipeline.sh       # Full pipeline: setup → train → extract
├── config/
│   ├── cfa.yaml
│   ├── cflow.yaml
│   ├── cutpaste.yaml
│   ├── deepsvdd.yaml
│   ├── dinomaly.yaml
│   ├── draem.yaml
│   ├── fastflow.yaml
│   ├── patchcore.yaml
│   ├── rd4ad.yaml
│   └── stfpm.yaml
├── models/
│   ├── flow_models.py        # FastFlow, CFlow registry + builder functions
│   ├── kd_models.py          # RD4AD, STFPM registry
│   ├── memory_models.py      # CFA, PatchCore registry
│   ├── recon_models.py       # DRAEM, Dinomaly registry
│   └── radimagenet_utils.py  # RadImageNet weight loading
├── data/
│   └── dataset_conversion.py # NIfTI → PNG conversion
├── Deep-SVDD/                # setup + training
└── pytorch-cutpaste/         # setup + training
```
