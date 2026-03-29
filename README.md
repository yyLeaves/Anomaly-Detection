# Post-Processing Pipeline — MR-OOD Anomaly Detection

Post-processing and evaluation pipeline for binary anomaly prediction masks produced by `ood-train`. Applies body masking, morphological filtering, and a 3D persistence filter, then computes pixel-, slice-, and patient-level metrics.

---

## Pipeline Overview

| Repository | Role |
| :--- | :--- |
| `../OOD-Data-Preprocessing` | Dataset preparation (NIfTI → PNG, body mask generation) |
| `../ood-train` | Model training and anomaly map / prediction mask extraction |
| **this repo** | Post-processing, evaluation, and visualization of model outputs |
| `../mp_visualizations` | Additional analysis notebooks and presentation figures |

---

## Input Structure

Raw prediction masks from `ood-train`:

```
<extraction_output_root>/
  anomaly_maps/test/{good,Ungood}/img/    ← per-slice .npy anomaly scores
  prediction_masks/test/{good,Ungood}/img/ ← binary PNG masks (0 or 255)
```

Body masks and ground truth from `OOD-Data-Preprocessing`:

```
<dataset_root>/test/
  good/img/        ← source MR slices (PNG)
  good/bodymask/   ← binary body masks (PNG)
  Ungood/img/
  Ungood/bodymask/
  Ungood/label/    ← ground-truth anomaly masks (PNG)
```

---

## Output Structure

```
<output_root>/
  01_body_masked_png/           ← after body-mask application
  02_morphology_png/            ← after small-component filter + morphological closing
  03_consecutive_filtered_png/  ← final masks (3D persistence filter)
  volumes/
    raw/                        ← 3D NIfTI from raw prediction masks
    post_processed/             ← 3D NIfTI from post-processed masks
    ground_truth/               ← 3D NIfTI from ground-truth masks
  metrics/
    metrics_summary.json        ← pixel/slice/patient metrics (raw + post-processed)
```

---

## Post-Processing Stages

The pipeline applies five sequential stages to the raw binary prediction masks:

1. **Body masking** (`apply_bodymask.py`): Multiply prediction masks element-wise by the anatomical body mask to remove out-of-body detections.
2. **Component filtering** (`morphology/processor.py`): Binarize (threshold=0.5) and remove connected components smaller than **τ_area = 3 pixels**.
3. **Morphological closing** (`morphology/processor.py`): Dilation × N followed by erosion × N. Fills small intra-region gaps and smooths contours.
4. **3D stacking** (`morphology/stack_to_3d.py`): Stack 2D PNG slices into patient-wise 3D NIfTI volumes.
5. **3D persistence filter** (`filter_prediction_masks_consecutive.py`): Discard any 2D connected component that does not overlap with an anomaly region in at least one neighbouring slice.

Default morphology parameters (configurable via CLI or `config/morpho_val.yaml`):

```
--min-component-size 3      # τ_area: minimum CC size in pixels
--kernel-size 5             # structuring element size (must be odd)
--kernel-shape ellipse      # or rect
--dilate-iterations 1       # dilation passes per round
--erode-iterations 1        # erosion passes per round
--num-rounds 1              # number of (dilate+erode) rounds
```

---

## Evaluation

`evaluate_model_outputs.py` computes metrics at three granularities:

- **Pixel level**: precision, recall, Dice score, false negative rate, balanced accuracy (aggregated over all prediction–ground-truth pixel pairs).
- **Slice level**: each 2D slice classified as positive/negative; standard binary classification metrics.
- **Patient level**: mean positive fraction (α_mean) per patient. Patients classified as anomalous if α_mean ≥ threshold; metrics reported for multiple thresholds (default: 0.0, 0.02, 0.05, 0.1).

`main_pipeline.py` automatically evaluates both the raw input masks and the stage 02 output (`02_morphology_png`) and writes both to `metrics/metrics_summary.json`.

### Standalone Metrics

```bash
python evaluate_model_outputs.py \
  --prediction-dir post_process_outputs/02_morphology_png \
  --ground-truth-dir /path/to/dataset/test \
  --ground-truth-replace img:label \
  --mean-fraction-thresholds 0.0 0.02 0.05 0.1 \
  --output-json metrics.json
```

---

## Quick Start

### Full Pipeline

```bash
python main_pipeline.py \
  --input-dir /path/to/prediction_masks/test \
  --body-mask-dir /path/to/dataset/root \
  --output-root post_process_outputs \
  --path-replace prediction_masks:test \
  --path-replace img:bodymask \
  --ground-truth-dir /path/to/dataset/root/test \
  --metrics-mean-fraction-thresholds 0.0 0.02 0.05 0.1 \
  --skip-missing-body-mask
```

### Morphology Parameter Tuning

Edit `config/morpho_val.yaml` with your validation mask paths and parameter combinations, then run:

```bash
python morphology/tune_morpho.py
```

Results are saved to `reports/morphology_tuning/tuning_report.json`.

---

## Visualization

Scripts in `visualization/` can be run directly from that folder (they add the repo root to the Python path automatically).

| Script | Output |
| :--- | :--- |
| **`visualize_processed_anomaly_maps.py`** | Side-by-side comparison of original vs. body-masked anomaly maps |
| **`visualize_processed_prediction_masks.py`** | Raw, body-masked, and filtered prediction mask panels |
| **`visualize_anomaly_thresholded_outputs.py`** | Anomaly map next to its thresholded binary output |
| **`convert_to_bone_colormap.py`** | Convert NIfTI slices to bone-colormap PNGs for inspection |

Example:

```bash
python visualization/visualize_processed_prediction_masks.py \
  --raw-dir /path/to/prediction_masks/test \
  --masked-dir post_process_outputs/01_body_masked_png \
  --image-dir /path/to/dataset \
  --image-replace prediction_masks:test \
  --output-dir prediction_mask_comparisons
```

---

## Repository Structure

```
Post-Processing-Pipeline/
│
├── main_pipeline.py                        # End-to-end pipeline entrypoint
├── apply_bodymask.py                       # Stage 0: body mask application
├── postprocess_utils.py                    # Shared I/O and array utilities
├── filter_prediction_masks_consecutive.py  # Stage 4: 3D persistence filter
├── evaluate_model_outputs.py               # Pixel/slice/patient metrics
├── compute_pixel_metrics.py                # Per-slice metric primitives
│
├── morphology/                             # Stages 1–3
│   ├── processor.py                        # MorphologyProcessor, BatchProcessor
│   ├── stack_to_3d.py                      # BatchNIfTIStacker (2D PNG → 3D NIfTI)
│   ├── slice_metrics.py                    # Metric helpers
│   ├── pipeline_tuning.py                  # Tuning pipeline logic
│   ├── tune_morpho.py                      # Tuning entrypoint
│   ├── apply_morpho.py                     # Standalone batch apply
│   └── README.md                           # Detailed morphology documentation
│
├── visualization/                          # Report and presentation figures
│   ├── visualize_processed_anomaly_maps.py
│   ├── visualize_processed_prediction_masks.py
│   ├── visualize_anomaly_thresholded_outputs.py
│   └── convert_to_bone_colormap.py
│
├── config/
│   └── morpho_val.yaml                     # Morphology tuning configuration
│
├── results/                                # Report figures
├── requirements.txt
└── README.md
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `anomalib==2.2.0`, `torch==2.8.0`, `nibabel==5.3.2`, `opencv-python==4.8.1.78`, `scipy==1.10.1`, `scikit-learn`.
