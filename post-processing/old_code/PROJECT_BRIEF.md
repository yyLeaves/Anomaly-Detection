# Post-Processing-Pipeline Refactor Brief (MR-OOD)

## Goal
Refactor this repo for grading: make it clean, easy to follow, and reproducible. Keep only code used for the final post-processing + evaluation pipeline in the active structure.

## Repo relationships
- ood-train: anomaly extraction/training; produces anomaly maps / predictions consumed here
- OOD-Data-Preprocessing: data preparation
- mp_visualizations: plotting; copy/move only the plots actually used into this repo

## Non-negotiable constraints
1) Do not delete anything. Move unused/legacy code into /old_code using `git mv`.
2) First change ONLY `README.md` (no code edits, no new code).
3) After README is finalized, only move files/folders (no modifications).
4) Keep all visualization code used for report/presentation, but do not refactor it yet.

## What README must explain (start with anomaly extraction)
1) Anomaly extraction happens in `ood-train` (brief I/O expectations).
2) This repo: post-processing + evaluation on model outputs.
3) Visualization scripts included to reproduce figures.

## Post-processing pipeline (must match report)
- Body mask on anomaly map/predictions
- Connected components filtering: 8-connectivity; remove components with area < tau_area = 3 px
- Morphological closing: dilation then erosion; configurable (defaults match report)
- 3D volumetric consistency: overlap persistence rule; keep detections on slice z only if overlap with z-1 or z+1
- Optional 3D reconstruction for visualization/patient aggregation

## Evaluation concepts to mention
- Report metrics “Before” vs “After” post-processing
- Pixel: Dice (F1), Precision, Recall
- Slice: rule = anomalous if any pixel in final mask is positive; metrics F1, Balanced Accuracy
- Patient: mean positive fraction f_z = positive/body pixels; f_bar = mean over slices; flag if f_bar >= alpha_mean; report patient metrics

## Refactor action plan
1) Inventory files and identify “used” entrypoints and their imports.
2) Rewrite README.md accordingly.
3) Create /old_code and move everything unused there via `git mv`.
4) Consolidate visualization scripts into a clear folder (no code changes).
