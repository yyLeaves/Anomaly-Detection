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
    bodymask/*.png              

  valid/
    good/
      img/*.png
      label/*.png              
      bodymask/*.png          
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
- Lightning’s internal checkpointing still places files under `<log_dir>/cflow_logs/version_*/checkpoints/` (best by `val_loss` plus `last.ckpt` when validation runs).
- After training the script copies `last.ckpt` (and `best.ckpt` when available) to `<log_dir>/cflow/<dataset_name>_cflow/weights/` and dumps run metadata as `<log_dir>/cflow_logs/version_*/training_run_metadata.json`.

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

The default Lightning checkpointing policy applies (saves best `val_loss` and the latest weights inside `<log_dir>/fastflow_logs/version_*/checkpoints/`). If you prefer a custom location or metric, add a `ModelCheckpoint` callback in the script.
- CSV metrics land in `<log_dir>/fastflow_logs/version_*/metrics.csv`.
- Lightning still writes default checkpoints under `<log_dir>/fastflow_logs/version_*/checkpoints/` based on `val_loss` (when available).
- Additional convenience copies of the latest/best FastFlow weights are written to `<log_dir>/fastflow/<dataset_name>_fastflow/weights/`. The accompanying metadata JSON is stored as `<log_dir>/fastflow_logs/version_*/training_run_metadata.json`.

## RadImageNet Weight Loading

Both scripts reuse `radimagenet_utils.py` to remap RadImageNet state dicts to standard torchvision ResNet module names before loading them into the anomalib models. Provide the `.pt` file via `--radimagenet_ckpt` or skip the flag to train from scratch.

## Notes

- GPU selection is index based. FastFlow accepts `--gpu_ids` for multi-GPU (`--gpu_ids 0,1`) or CPU-only runs (`--gpu_ids cpu`). CFlow continues to use `--gpu` with `-1` forcing CPU.
- Lightning’s default loggers create CSV metrics inside `<log_dir>/<model>_logs/version_*/metrics.csv` for convenient inspection.
- FastFlow’s evaluator requires masks only when pixel metrics are enabled; absence of `label/` directories will disable pixel metrics automatically inside anomalib.

For more advanced usage (additional callbacks, mixed precision, distributed training), treat these scripts as starting points and extend them with standard Lightning components.

## Exporting FastFlow Anomaly Maps

After training, use `extract_fastflow.py` to run inference on a split and export per-image anomaly maps as `.npy` files:

```
python extract_fastflow.py \
  --data_root /local/scratch/koepchen/synth23_pelvis_v8_png \
  --checkpoint fastflow/synth23_pelvis_v8_png_fastflow/weights/best.ckpt \
  --output_dir extracted_anomaly_maps_fastflow \
  --split test \
  --batch_size 8 \
  --gpu 0 \
  --map_size 224
```

The script reconstructs the dataset layout under `--output_dir`, saving `<image>_anomaly_map.npy` beneath `anomaly_maps/<split>/.../img/` and matching prediction masks as PNGs. Use `--map_size` to control the exported spatial resolution (default 224) and `--metadata` if you need a manifest of the generated files.

## Post-processing FastFlow Outputs

Run `apply_bodymask_fastflow.py` to multiply anomaly maps (and optional prediction masks) by the anatomical body masks:

```
python apply_bodymask_fastflow.py \
  --anomaly-dir extracted_anomaly_maps_fastflow/anomaly_maps \
  --body-mask-dir /local/scratch/koepchen/synth23_pelvis_v8_png \
  --output-dir postprocessed_anomaly_maps \
  --path-replace anomaly_maps:test \
  --path-replace img:bodymask \
  --prediction-mask-dir prediction_masks_fastflow
```

- `--anomaly-dir`: Root of the exported anomaly maps (respects their relative subfolders).  
- `--body-mask-dir`: Root of the dataset containing the body-mask folders; combine with `--path-replace` to swap components (e.g. `anomaly_maps` → `test` and `img` → `bodymask`).  
- `--output-dir`: Destination for masked anomaly maps; set `--skip-missing` to ignore slices without masks.  
- `--prediction-mask-dir`: Applies the body mask to existing prediction masks and writes the result to this directory. Adjust `--raw-prediction-dir` if the raw masks live outside the default `prediction_masks/` tree.

A summary line reports how many anomaly maps were processed and where masked maps (and prediction masks, if requested) were written.

If you already have masked anomaly maps, you can rerun the command with `--masked-dir postprocessed_anomaly_maps/test` and omit `--body-mask-dir` to regenerate prediction masks only.

## Visualising Processed Anomaly Maps

When you only need the qualitative before/after views, use `visualize_processed_anomaly_maps.py`:

```
python visualize_processed_anomaly_maps.py \
  --anomaly-dir extracted_anomaly_maps_fastflow/anomaly_maps \
  --masked-dir postprocessed_anomaly_maps/test \
  --image-dir /local/scratch/koepchen/synth23_pelvis_v8_png \
  --image-replace anomaly_maps:test \
  --comparison-dir comparisons_fastflow \
  --overlay-dir overlays_fastflow \
  --comparison-cmap magma \
  --overlay-alpha 0.6
```

This CLI handles the visualisation (comparison panels and overlays) without modifying the anomaly maps or prediction masks.

## Exporting Prediction Masks

If you need binary prediction masks derived from the masked anomaly maps, reuse the pixel threshold stored in the FastFlow checkpoint:

```
python export_prediction_masks.py \
  --masked-dir postprocessed_anomaly_maps/test \
  --checkpoint fastflow/synth23_pelvis_v8_png_fastflow/weights/best-v2.ckpt \
  --output-dir prediction_masks_fastflow
```

This helper thresholds masked anomaly maps using the learned pixel threshold stored in the checkpoint (useful when you are not reusing `apply_bodymask_fastflow.py` for prediction masks).

## Visualising Prediction Masks

`visualize_processed_prediction_masks.py` compares the raw prediction masks with their body-masked counterparts. Example:

```
python visualize_processed_prediction_masks.py \
  --raw-dir extracted_anomaly_maps_fastflow/prediction_masks/test \
  --masked-dir prediction_masks_fastflow/test \
  --image-dir /local/scratch/koepchen/synth23_pelvis_v8_png \
  --image-replace prediction_masks:test \
  --output-dir prediction_mask_comparisons \
  --cmap magma \
  --overlay-alpha 0.6
```

Provide `--image-dir`/`--image-replace` if you want overlay panels; otherwise the script produces side-by-side PNGs showing raw vs. body-masked masks without touching the underlying data.
