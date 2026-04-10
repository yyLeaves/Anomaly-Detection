# Medical Image Data Preprocessing for Anomaly Detection (Synthrad 2023 Pelvis)

This repository contains the modularized data preprocessing pipeline for the **Synthrad 2023 Pelvis** dataset. The pipeline transforms 3D CT/MR volumes into processed 2D slices, performing intensity normalization, automated artifact segmentation, and standardized geometric transformations.

---

## Modular Component Overview

The logic is split between **Driver Scripts** (high-level orchestration) and **Utility Modules** (the processing engine).

### 1. Main Dataset Processing Scripts (Drivers)
These scripts handle the entire pipeline from loading to final export. Each exists in a **sc_** (Single-Center) and **mc_** (Multi-Center) version.

| Script Name | Output Format | Input Slice Channels | Output Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **{sc/mc}_dataset_processing_png.py** | **PNG (.png)** | 1 Slice (RGB) | **(224, 224)** | Extracts the current 2D slice, maps it to **'bone'** colormap, and saves as a 3-channel PNG. |
| **{sc/mc}_dataset_processing_nifti_rep.py** | **NIfTI (.nii)** | 1 Slice (Replicated) | **(224, 224)** | Extracts the current slice and replicates it across 3 channels into a NIfTI file. |
| **{sc/mc}_dataset_processing_nifti_con.py** | **NIfTI (.nii)** | 3 Consecutive Slices | **(224, 224)** | Extracts **(prev, current, next)** slices to form a 3-channel NIfTI, capturing spatial context. |

### 2. Utility Modules (`utils/`)
| Module | Core Responsibility |
| :--- | :--- |
| **artifact_detector.py** | Houses the `MetalArtifactDetector` class for HU-based thresholding and MR refinement. |
| **processing_utils.py** | Manages the geometry pipeline: padding, resizing, cropping, and min-max normalization. |
| **io_utils.py** | Standardized saving/loading for NIfTI and PNG to ensure consistent data types. |
| **path_utils.py** | Handles the automated creation of the `img`, `label`, and `bodymask` subfolder structures. |
| **label_generator.py** | Parses raw OOD text files into structured JSON annotations for implants and other anomalies. |


### 3. Annotation & Label Mapping (`labels/`)
The Out-of-Distribution (OOD) metadata is consolidated into two JSON files:

* **`labels_implant.json`**: Contains slice-level annotations specifically for **Type 1 (Implants)**.
* **`labels_others.json`**: Contains annotations for **all other anomaly categories** (Types 2–7).

Each entry provides the `mr_start/end` and `ct_start/end` slice indices, along with the `body_part` (Brain or Pelvis) classification.

---

## The Preprocessing Pipeline

All driver scripts follow a standardized 5-step pipeline:

1. **Loading**: Pairs MR, CT, and Mask volumes for a specific Patient ID.
2. **Normalization**: Applies a body mask to MR data and performs min-max normalization ($0 \rightarrow 255$) for consistent intensity ranges.
3. **Anomaly Detection**: For "Ungood" (abnormal) scans, the system calculates a dynamic threshold ($\tau$) based on CT Hounsfield Units to isolate implants, which is then refined using the MR signal.
4. **Geometric Standardization**: 
    * **Square Padding**: Slices are center-padded to a 1:1 aspect ratio.
    * **Resizing**: Slices are interpolated to a base size of **240x240**.
    * **Center Crop**: A final crop to **224x224** is performed to remove edge artifacts.
5. **Structured Export**: Slices are sorted into `train`, `valid`, or `test` directories. A specialized **"Whole Patient"** export is performed for test cases to allow for full volumetric evaluation.

---

## Usage

Run the driver script corresponding to your required center type and output format. 

### Single-Center PNG Example:
```bash
python sc_dataset_processing_png.py \
  --dir_pelvis /path/to/raw/data/Task1/pelvis \
  --dir_output /path/to/save/png_dataset
```
