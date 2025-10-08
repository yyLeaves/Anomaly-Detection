# OOD Flow Training Utilities

This repository contains training entry points for flow-based anomaly detection models built on [anomalib](https://github.com/openvinotoolkit/anomalib). They are tailored for datasets organised with separate `img/` and `label/` folders (PNG images plus optional pixel-level masks) and focus on two architectures:

- **CFlow** (`train_cflow.py`)
- **FastFlow** (`train_fastflow.py`)

Both scripts assume a RadImageNet-pretrained backbone but can be used with other backbones exposed by anomalib.

## Dataset Layout

Each dataset root is expected to follow the anomalib folder convention:

```
<data_root>/
  train/
    good/*.png
    bodymask/*.png              # optional whole-body masks aligned with training frames

  valid/
    good/
      img/*.png
      label/*.png               # optional anomaly masks
      bodymask/*.png            # optional whole-body masks
    Ungood/
      img/*.png
      label/*.png
      bodymask/*.png

  test/
    good/
      img/*.png
      label/*.png
      bodymask/*.png
    Ungood/
      img/*.png
      label/*.png
      bodymask/*.png
```

If `valid/` exists it is used for evaluation; otherwise the scripts fall back to `test/`. Masks in `label/` are optional but enable pixel-level metrics during FastFlow training. Whole-body masks in `bodymask/` are ignored by the minimal trainers but kept in the layout for compatibility with the fuller pipelines.

## Common Arguments

Both scripts share a similar CLI:

- `--data_root`: Path to the dataset root (default: `/local/scratch/koepchen/synth23_pelvis_v6_png`).
- `--log_dir`: Base directory where Lightning logs and checkpoints are stored (default: `/home/user/koepchen/OOD_tests`).
- `--backbone`: Backbone identifier. Values like `radimagenet_resnet50`, `radimagenet_resnet18`, or any backbone supported by the respective model.
- `--radimagenet_ckpt`: Optional path to a RadImageNet `.pt` checkpoint to seed the backbone weights.
- `--epochs`: Maximum training epochs.
- `--batch_size`: Training and evaluation batch size.
- `--gpu_ids`: (FastFlow only) Comma separated GPU indices (e.g. `0,1`). Use `cpu` to force CPU execution.
- `--gpu`: GPU index to use (`-1` forces CPU). For FastFlow this flag is deprecated in favour of `--gpu_ids`.

## Training CFlow on PNGs

`train_cflow.py` runs CFlow with PNG inputs using anomalib’s `Folder` datamodule.

Example command:

```
python new_train_cflow.py \
  --data_root /local/scratch/koepchen/synth23_pelvis_v6_png/synth23_pelvis_v6_png \
  --backbone radimagenet_resnet50 \
  --radimagenet_ckpt RadImageNet_pytorch/ResNet50.pt \
  --epochs 10 \
  --batch_size 16 \
  --gpu 0
```

- CSV metrics land in `<log_dir>/cflow_logs/version_*/metrics.csv`.
- Lightning’s default checkpointing continues to write into `<log_dir>/cflow_logs/version_*/checkpoints/` when validation is configured.
- A dedicated `ModelCheckpoint` mirrors the reference pipeline by keeping `best.ckpt` (lowest `train_loss_step`) and `last.ckpt` in both `<log_dir>/cflow/<dataset_name>_cflow/weights/` (global log root) and `<repo_root>/cflow/<dataset_name>_cflow/weights/`. Run metadata is saved to `<log_dir>/cflow_logs/version_*/training_run_metadata.json` and copied to `<repo_root>/cflow_logs/version_*/`.

## Training FastFlow on PNGs with Pixel Metrics

`train_fastflow.py` provides a streamlined FastFlow trainer. It automatically looks for anomaly masks in `label/` directories to enable pixel-level evaluation.

Example command:

```
python train_fastflow.py \
  --data_root /local/scratch/koepchen/synth23_pelvis_v6_png/synth23_pelvis_v6_png \
  --backbone radimagenet_resnet50 \
  --radimagenet_ckpt RadImageNet_pytorch/ResNet50.pt \
  --epochs 50 \
  --batch_size 32 \
  --gpu_ids 0,1
```

- The default Lightning checkpointing policy applies (saves best `val_loss` and the latest weights inside `<log_dir>/fastflow_logs/version_*/checkpoints/`). If you prefer a custom location or metric, add or edit the `ModelCheckpoint` callback in the script.
- CSV metrics land in `<log_dir>/fastflow_logs/version_*/metrics.csv`.
- To mirror the fuller pipeline, an explicit `ModelCheckpoint` stores `best.ckpt` (lowest `train_loss_step`) and `last.ckpt` under both `<log_dir>/fastflow/<dataset_name>_fastflow/weights/` and `<repo_root>/fastflow/<dataset_name>_fastflow/weights/`. Metadata JSONs are kept at `<log_dir>/fastflow_logs/version_*/training_run_metadata.json` and mirrored to `<repo_root>/fastflow_logs/version_*/`.

## RadImageNet Weight Loading

Both scripts reuse `radimagenet_utils.py` to remap RadImageNet state dicts to standard torchvision ResNet module names before loading them into the anomalib models. Provide the `.pt` file via `--radimagenet_ckpt` or skip the flag to train from scratch.

## Exporting FastFlow Anomaly Maps

After training, use `test_fastflow.py` to run inference on a split and export per-image anomaly maps as `.npy` files:

```
python test_fastflow.py \
  --data_root /local/scratch/koepchen/synth23_pelvis_v8_png \
  --checkpoint fastflow/synth23_pelvis_v8_png_fastflow/weights/best.ckpt \
  --output_dir ./fastflow_anomaly_maps \
  --split test \
  --batch_size 8 \
  --gpu 0
```

The script reuses the training helpers to construct the datamodule, loads the specified checkpoint, and saves `<image_stem>_anomaly_map.npy` files under `--output_dir/anomaly_maps/` (with PNG prediction masks under `--output_dir/prediction_masks/`). Each result is also mirrored into `maps_fastflow/<split>/anomaly_maps/` and `maps_fastflow/<split>/prediction_masks/` for easy downstream use.

## Post-processing FastFlow Outputs

After inference, apply the anatomical body masks to the exported FastFlow anomaly maps before running downstream metrics. Use `fastflow_postprocess.py`:

```
python fastflow_postprocess.py \
  --anomaly-dir /path/to/raw_anomaly_maps \
  --body-mask-dir /local/scratch/koepchen/synth23_pelvis_v8_png/test/Ungood/bodymask \
  --output-dir ./postprocessed_anomaly_maps \
  --path-replace img:bodymask
```

- Supply the directory that contains the raw anomaly maps (`--anomaly-dir`) and the body-mask tree (`--body-mask-dir`).  
- `--path-replace` rewrites path components when body masks live in a sibling folder (e.g. replace `img` with `bodymask`). Pass it multiple times if you need several substitutions.
- The script preserves the relative layout while writing masked maps to `--output-dir`. By default it stops if a body mask is missing; add `--skip-missing` to emit warnings instead.
- Provide `--image-dir` (and optional `--image-replace`) together with `--comparison-dir` to generate side-by-side panels showing the original image, the raw anomaly map, and the masked anomaly map using the chosen colormap.

## Notes

- GPU selection is index based. FastFlow accepts `--gpu_ids` for multi-GPU (`--gpu_ids 0,1`) or CPU-only runs (`--gpu_ids cpu`). CFlow continues to use `--gpu` with `-1` forcing CPU.
- Lightning’s default loggers create CSV metrics inside `<log_dir>/<model>_logs/version_*/metrics.csv` for convenient inspection.
- FastFlow’s evaluator requires masks only when pixel metrics are enabled; absence of `label/` directories will disable pixel metrics automatically inside anomalib.

For more advanced usage (additional callbacks, mixed precision, distributed training), treat these scripts as starting points and extend them with standard Lightning components.
