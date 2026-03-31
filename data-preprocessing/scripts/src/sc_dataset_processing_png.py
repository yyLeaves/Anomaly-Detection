import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.artifact_detector import MetalArtifactDetector
from utils.path_utils import create_output_dirs
from utils.io_utils import save_png, get_ids_from_ungood_test_folder
from utils.processing_utils import (
    center_pad_single_slice,
    center_pad_single_slice_by_params,
    resize_image,
    center_crop,
    load_scan
)

# ================================================================
# CONFIGURATION
# ================================================================
DELTA = 200
SLICE_INDEX_START_NORMAL = 25
SLICE_INDEX_END_NORMAL = -20
TARGET_SIZE = (240, 240)
SEED = 24
random.seed(SEED)
np.random.seed(SEED)

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PNG pelvis preprocessing")
    parser.add_argument("--dir_pelvis", type=str, required=True, help="Path to raw NIfTI data")
    parser.add_argument("--dir_output", type=str, required=True, help="Path to save processed PNGs")
    return parser.parse_args()

# ================================================================
# SAVE FUNCTIONS
# ================================================================
def save_train_slice(slice_img, slice_body_mask, output_dir, split, subset, id_, i):
    """Save structure for TRAIN (no label folder)."""
    path_img = os.path.join(output_dir, split, subset, f"{id_}_{i}.png")
    path_bodymask = os.path.join(output_dir, split, "bodymask", f"{id_}_{i}.png")
    save_png(slice_img, path_img, cmap="bone")
    save_png(slice_body_mask, path_bodymask, cmap="binary")

def save_eval_slice(slice_img, slice_mask, slice_body_mask, output_dir, split, subset, id_, i):
    """Save structure for VALID and TEST (img/label/bodymask)."""
    base_path = os.path.join(output_dir, split, subset)
    save_png(slice_img, os.path.join(base_path, "img", f"{id_}_{i}.png"), cmap="bone")
    save_png(slice_mask, os.path.join(base_path, "label", f"{id_}_{i}.png"), cmap="binary")
    save_png(slice_body_mask, os.path.join(base_path, "bodymask", f"{id_}_{i}.png"), cmap="binary")

# ================================================================
# CORE PROCESSING
# ================================================================
def process_slices(
    mr_norm, body_mask_vol, id_, split, subset,
    output_dir, start_idx=SLICE_INDEX_START_NORMAL, end_offset=SLICE_INDEX_END_NORMAL,
    mask_vol=None, abnormal_slices=None
):
    """Process and save all slices (good or Ungood) for a scan."""
    slices = mr_norm.shape[2]
    slice_indices = (
        range(start_idx, slices + end_offset)
        if abnormal_slices is None else abnormal_slices
    )

    for i in tqdm(slice_indices, desc=f"{id_}-{split}-{subset}"):
        slice_img = mr_norm[:, :, i]
        slice_body_mask = body_mask_vol[:, :, i]
        slice_mask = mask_vol[:, :, i] if mask_vol is not None else np.zeros_like(slice_body_mask)

        # Center pad to square
        slice_img_centered, (pad_h, pad_w) = center_pad_single_slice(slice_img)
        slice_body_mask_centered = center_pad_single_slice_by_params(slice_body_mask, pad_h, pad_w)
        slice_mask_centered = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)

        # Resize and Crop to 224x224 (implied by default center_crop)
        slice_img_cropped = center_crop(resize_image(slice_img_centered, target_size=TARGET_SIZE))
        slice_body_mask_cropped = center_crop(resize_image(slice_body_mask_centered, target_size=TARGET_SIZE))
        slice_mask_cropped = center_crop(resize_image(slice_mask_centered, target_size=TARGET_SIZE))

        # Skip tiny abnormal masks
        if mask_vol is not None and slice_mask_cropped.sum() < 3:
            continue

        if split == "train":
            save_train_slice(slice_img_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i)
        else:
            save_eval_slice(slice_img_cropped, slice_mask_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i)

def process_good_scans(det, ids, split, dir_pelvis, dir_output):
    for id_ in ids:
        _, mr_norm, _, body_mask_vol = load_scan(dir_pelvis, det, id_)
        process_slices(mr_norm, body_mask_vol, id_, split, "good", dir_output)

def process_ungood_scans(det, ids, split, dir_pelvis, dir_output, anomaly_range):
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_)
        slices = mr_norm.shape[2]
        abnormal_slices = list(range(anomaly_range[id_][0], anomaly_range[id_][-1]))

        # Compute HU-based mask via Detector
        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        
        tau = min(scan_value - DELTA, 2000)
        mask_vol = (ct >= tau).astype(np.uint8)
        
        # Refine and Morphological Postprocessing
        mask_ref = det.refine_mask_with_mr(mask_vol, mr, lo_diff=5, up_diff=10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, min_area_for_smooth=50, slice_axis=2)

        # Filter indices to safe ranges
        abnormal_slices = [idx for idx in abnormal_slices if 15 <= idx < slices - 15]
        process_slices(mr_norm, body_mask_vol, id_, split, "Ungood", dir_output, mask_vol=mask_ref, abnormal_slices=abnormal_slices)

# ================================================================
# PATIENT-WISE EXPORT
# ================================================================
def export_full_anomalous_cases_png(det, ids, dir_pelvis, dir_output, anomaly_range):
    """Save all slices for test patients in structured folders."""
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_)
        slices = mr_norm.shape[2]
        start_idx, end_idx = 10, slices - 5
        
        if end_idx <= start_idx: continue

        # Recalculate Refined Mask for patient-wise
        abn_start, abn_end = anomaly_range.get(id_, (start_idx, end_idx))
        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], range(abn_start, abn_end)).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        
        tau = min(scan_value - DELTA, 2000)
        mask_vol = (ct >= tau).astype(np.uint8)
        mask_ref = det.refine_mask_with_mr(mask_vol, mr, lo_diff=5, up_diff=10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, min_area_for_smooth=50, slice_axis=2)

        base_path = os.path.join(dir_output, "test", "Ungood_whole_patient_scans")
        
        for i in tqdm(range(start_idx, end_idx), desc=f"Patient-wise {id_}"):
            # Re-run same padding/resizing logic
            s_img_c, (ph, pw) = center_pad_single_slice(mr_norm[:, :, i])
            s_bm_c = center_pad_single_slice_by_params(body_mask_vol[:, :, i], ph, pw)
            s_m_c = center_pad_single_slice_by_params(mask_ref[:, :, i], ph, pw)

            img_crop = center_crop(resize_image(s_img_c, target_size=TARGET_SIZE))
            bm_crop = center_crop(resize_image(s_bm_c, target_size=TARGET_SIZE))
            m_crop = center_crop(resize_image(s_m_c, target_size=TARGET_SIZE))

            save_png(img_crop, os.path.join(base_path, "img", f"{id_}_{i}.png"), cmap="bone")
            save_png(m_crop, os.path.join(base_path, "label", f"{id_}_{i}.png"), cmap="binary")
            save_png(bm_crop, os.path.join(base_path, "bodymask", f"{id_}_{i}.png"), cmap="binary")

# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    create_output_dirs(args.dir_output)

    # 1. Loading IDs and Label Ranges
    excel_path = os.path.join(args.dir_pelvis, "overview", "1_pelvis_train.xlsx")
    df_overview = pd.read_excel(excel_path, sheet_name="MR")
    ids_all = [i for i in df_overview["ID"].tolist() if i.startswith("1PA")]

    # Load JSON Labels (paths simplified for example)
    with open("/home/user/jverhoek/sct-ood-dataset/labels/labels_implant.json") as f:
        data_implant = json.load(f)
    
    # Process Anomaly Ranges
    df_labels = pd.DataFrame([{"id": k, **v} for item in data_implant['type1'] for k, v in item.items()])
    df_labels = df_labels.dropna().query("body_part == 'pelvis'")
    df_labels["anomaly_start"] = df_labels[["ct_start", "mr_start"]].min(axis=1)
    df_labels["anomaly_end"] = df_labels[["ct_end", "mr_end"]].max(axis=1)
    
    anomaly_range = {row.id: (int(row.anomaly_start), int(row.anomaly_end)) for row in df_labels.itertuples()}

    ids_abnormal_all = [i for i in df_labels['id'].tolist() if not i.startswith("1PC")]
    # Remove known problematic scans
    for bad_id in ['1PA030', '1PA170']:
        if bad_id in ids_abnormal_all: ids_abnormal_all.remove(bad_id)

    # 2. Train/Val/Test Splitting
    ids_normal_all = sorted(set(ids_all) - set(ids_abnormal_all))
    random.shuffle(ids_normal_all)
    random.shuffle(ids_abnormal_all)

    # Simple index-based split
    val_n, test_n = int(len(ids_normal_all)*0.1), int(len(ids_normal_all)*0.2)
    ids_norm_val = ids_normal_all[:val_n]
    ids_norm_test = ids_normal_all[val_n:val_n+test_n]
    ids_norm_train = ids_normal_all[val_n+test_n:]

    val_a, test_a = max(4, int(len(ids_abnormal_all)*0.1)), max(1, int(len(ids_abnormal_all)*0.2))
    ids_abn_val = ids_abnormal_all[:val_a]
    ids_abn_test = ids_abnormal_all[val_a:]

    # 3. Processing
    det = MetalArtifactDetector()

    logger.info("Processing Splits...")
    process_good_scans(det, ids_norm_train, "train", args.dir_pelvis, args.dir_output)
    process_good_scans(det, ids_norm_val, "valid", args.dir_pelvis, args.dir_output)
    process_ungood_scans(det, ids_abn_val, "valid", args.dir_pelvis, args.dir_output, anomaly_range)
    process_good_scans(det, ids_norm_test, "test", args.dir_pelvis, args.dir_output)
    process_ungood_scans(det, ids_abn_test, "test", args.dir_pelvis, args.dir_output, anomaly_range)

    # 4. Patient-wise Test Export
    ids_effective = sorted(get_ids_from_ungood_test_folder(args.dir_output).intersection(ids_abn_test))
    export_full_anomalous_cases_png(det, ids_effective, args.dir_pelvis, args.dir_output, anomaly_range)

    logger.info("✅ Dataset Creation Complete.")