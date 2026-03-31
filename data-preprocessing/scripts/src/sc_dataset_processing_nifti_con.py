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
from utils.io_utils import save_np_to_nifti, get_ids_from_ungood_test_folder
from utils.processing_utils import (
    load_scan,
    extract_3ch_slice_con,  
    center_pad_single_slice,
    center_pad_single_slice_by_params,
    resize_image,
    center_crop
)

# ================================================================
# CONFIGURATION
# ================================================================
DELTA = 200
THRESH_MR_MASK = 0.1
SLICE_INDEX_START_NORMAL = 25
SLICE_INDEX_END_NORMAL = -20
TARGET_SIZE = (240, 240)
TARGET_SIZE_CROP = (224, 224)
SEED = 24
random.seed(SEED)
np.random.seed(SEED)

# ================================================================
# HELPER FUNCTIONS
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Consecutive 3-channel NIfTI pelvis preprocessing")
    parser.add_argument("--dir_pelvis", type=str, required=True, help="Path to raw NIfTI data")
    parser.add_argument("--dir_output", type=str, required=True, help="Path to save processed NIfTIs")
    return parser.parse_args()

# ================================================================
# SAVE FUNCTIONS
# ================================================================
def save_train_slice(slice_imgs_nifti, slice_body_mask_cropped, output_dir, subset, id_, i):
    """Save structure for TRAIN (no label folder)."""
    save_np_to_nifti(slice_imgs_nifti, os.path.join(output_dir, "train", subset, f"{id_}_{i}.nii"))
    save_np_to_nifti(slice_body_mask_cropped, os.path.join(output_dir, "train", "bodymask", f"{id_}_{i}.nii"))

def save_eval_slice(slice_imgs_nifti, slice_mask_cropped, slice_body_mask_cropped, output_dir, split, subset, id_, i):
    """Save structure for VALID and TEST (img/label/bodymask subfolders)."""
    base_path = os.path.join(output_dir, split, subset)
    save_np_to_nifti(slice_imgs_nifti, os.path.join(base_path, "img", f"{id_}_{i}.nii"))
    save_np_to_nifti(slice_mask_cropped, os.path.join(base_path, "label", f"{id_}_{i}.nii"))
    save_np_to_nifti(slice_body_mask_cropped, os.path.join(base_path, "bodymask", f"{id_}_{i}.nii"))

# ================================================================
# MAIN PROCESSING FUNCTION
# ================================================================
def process_slices(
    mr_norm, body_mask_vol, id_, split, subset,
    output_dir, start_idx=SLICE_INDEX_START_NORMAL, end_offset=SLICE_INDEX_END_NORMAL,
    mask_vol=None, abnormal_slices=None
):
    """Process slices using consecutive (i-1, i, i+1) context into 3-channel NIfTI."""
    slices = mr_norm.shape[2]
    slice_indices = (
        range(start_idx, slices + end_offset)
        if abnormal_slices is None else abnormal_slices
    )

    for i in tqdm(slice_indices, desc=f"{id_}-{split}-{subset}"):
        # 1. Extract Consecutive slices (i-1, i, i+1)
        slice_imgs = extract_3ch_slice_con(mr_norm, i, start_idx, end_offset)
        slice_body_mask = body_mask_vol[:, :, i]
        slice_mask = mask_vol[:, :, i] if mask_vol is not None else np.zeros_like(slice_body_mask)

        # 2. Geometry: Padding from the center slice
        _, (pad_h, pad_w) = center_pad_single_slice(slice_imgs[:, :, 1])

        # 3. Process 3 Context Channels
        processed_channels = []
        for c in range(3):
            p = center_pad_single_slice_by_params(slice_imgs[:, :, c], pad_h, pad_w)
            r = resize_image(p, target_size=TARGET_SIZE)
            processed_channels.append(center_crop(r, target_size=TARGET_SIZE_CROP))
        
        # 4D NIfTI stack: (H, W, 1, 3)
        slice_imgs_nifti = np.expand_dims(np.stack(processed_channels, axis=-1), axis=2)

        # 4. Process Masks
        bm_p = center_pad_single_slice_by_params(slice_body_mask, pad_h, pad_w)
        m_p = center_pad_single_slice_by_params(slice_mask, pad_h, pad_w)
        
        bm_final = center_crop(resize_image(bm_p, target_size=TARGET_SIZE), target_size=TARGET_SIZE_CROP)
        mask_final = center_crop(resize_image(m_p, target_size=TARGET_SIZE), target_size=TARGET_SIZE_CROP)

        if mask_vol is not None and mask_final.sum() < 3:
            continue

        if split == "train":
            save_train_slice(slice_imgs_nifti, bm_final, output_dir, subset, id_, i)
        else:
            save_eval_slice(slice_imgs_nifti, mask_final, bm_final, output_dir, split, subset, id_, i)

# ================================================================
# SPLIT HANDLERS
# ================================================================
def process_good_scans(det, ids, split, dir_pelvis, dir_output):
    for id_ in ids:
        _, mr_norm, _, body_mask_vol = load_scan(dir_pelvis, det, id_, THRESH_MR_MASK)
        process_slices(mr_norm, body_mask_vol, id_, split, "good", dir_output)

def process_ungood_scans(det, ids, split, dir_pelvis, dir_output, anomaly_range):
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_, THRESH_MR_MASK)
        slices = mr_norm.shape[2]
        ab_start, ab_end = anomaly_range[id_]
        abnormal_indices = list(range(ab_start, ab_end))

        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_indices).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        tau = min(scan_value - DELTA, 2000)

        mask_vol = (ct >= tau).astype(np.uint8)
        mask_ref = det.refine_mask_with_mr(mask_vol, mr, lo_diff=5, up_diff=10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, min_area_for_smooth=50, slice_axis=2)

        safe_slices = [i for i in abnormal_indices if 15 <= i < slices - 15]
        process_slices(mr_norm, body_mask_vol, id_, split, "Ungood", dir_output, mask_vol=mask_ref, abnormal_slices=safe_slices)

# ================================================================
# PATIENT-WISE EXPORT
# ================================================================
def export_full_anomalous_cases_nifti(det, ids, dir_pelvis, dir_output, anomaly_range):
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_, THRESH_MR_MASK)
        slices = mr_norm.shape[2]
        start_idx, end_idx = 10, slices - 5
        if end_idx <= start_idx: continue

        ab_start, ab_end = anomaly_range.get(id_, (start_idx, end_idx))
        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], list(range(ab_start, ab_end))).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        tau = min(scan_value - DELTA, 2000)
        mask_ref = det.refine_mask_with_mr((ct >= tau).astype(np.uint8), mr, 5, 10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, min_area_for_smooth=50, slice_axis=2)

        base_path = os.path.join(dir_output, "test", "Ungood_whole_patient_scans")
        for i in tqdm(range(start_idx, end_idx), desc=f"Patient-wise NIfTI {id_}"):
            slice_imgs = extract_3ch_slice_con(mr_norm, i, start_idx, -5)
            _, (ph, pw) = center_pad_single_slice(slice_imgs[:, :, 1])

            processed_ch = []
            for c in range(3):
                p = center_pad_single_slice_by_params(slice_imgs[:, :, c], ph, pw)
                processed_ch.append(center_crop(resize_image(p, TARGET_SIZE), TARGET_SIZE_CROP))
            
            slice_nifti = np.expand_dims(np.stack(processed_ch, axis=-1), axis=2)
            bm_final = center_crop(resize_image(center_pad_single_slice_by_params(body_mask_vol[:, :, i], ph, pw), TARGET_SIZE), TARGET_SIZE_CROP)
            m_final = center_crop(resize_image(center_pad_single_slice_by_params(mask_ref[:, :, i], ph, pw), TARGET_SIZE), TARGET_SIZE_CROP)

            save_np_to_nifti(slice_nifti, os.path.join(base_path, "img", f"{id_}_{i}.nii"))
            save_np_to_nifti(m_final, os.path.join(base_path, "label", f"{id_}_{i}.nii"))
            save_np_to_nifti(bm_final, os.path.join(base_path, "bodymask", f"{id_}_{i}.nii"))

# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    create_output_dirs(args.dir_output)

    # 1. Loading IDs
    excel_path = os.path.join(args.dir_pelvis, "overview", "1_pelvis_train.xlsx")
    df_overview = pd.read_excel(excel_path, sheet_name="MR")
    ids_all = [i for i in df_overview["ID"].tolist() if i.startswith("1PA")]

    # Load JSON Labels
    with open("/home/user/jverhoek/sct-ood-dataset/labels/labels_implant.json") as f:
        data_implant = json.load(f)
    
    df_labels = pd.DataFrame([{"id": k, **v} for item in data_implant['type1'] for k, v in item.items()])
    df_labels = df_labels.dropna().query("body_part == 'pelvis'")
    
    anomaly_range = {
        row.id: (int(min(row.ct_start, row.mr_start)), int(max(row.ct_end, row.mr_end))) 
        for row in df_labels.itertuples()
    }

    ids_abnormal_all = [i for i in df_labels['id'].tolist() if not i.startswith("1PC")]
    for bad_id in ['1PA030', '1PA170']:
        if bad_id in ids_abnormal_all: ids_abnormal_all.remove(bad_id)

    ids_normal_all = sorted(set(ids_all) - set(ids_abnormal_all))
    random.shuffle(ids_normal_all)
    random.shuffle(ids_abnormal_all)

    # 2. Splits
    val_n, test_n = int(len(ids_normal_all)*0.1), int(len(ids_normal_all)*0.2)
    ids_norm_val, ids_norm_test, ids_norm_train = ids_normal_all[:val_n], ids_normal_all[val_n:val_n+test_n], ids_normal_all[val_n+test_n:]

    val_a = max(4, int(len(ids_abnormal_all)*0.1))
    ids_abn_val, ids_abn_test = ids_abnormal_all[:val_a], ids_abnormal_all[val_a:]

    # 3. Processing
    det = MetalArtifactDetector()
    logger.info("Starting NIfTI (Consecutive) Processing...")

    process_good_scans(det, ids_norm_train, "train", args.dir_pelvis, args.dir_output)
    process_good_scans(det, ids_norm_val, "valid", args.dir_pelvis, args.dir_output)
    process_ungood_scans(det, ids_abn_val, "valid", args.dir_pelvis, args.dir_output, anomaly_range)
    process_good_scans(det, ids_norm_test, "test", args.dir_pelvis, args.dir_output)
    process_ungood_scans(det, ids_abn_test, "test", args.dir_pelvis, args.dir_output, anomaly_range)

    # 4. Patient-wise Test Export
    ids_eff = sorted(get_ids_from_ungood_test_folder(args.dir_output).intersection(ids_abn_test))
    export_full_anomalous_cases_nifti(det, ids_eff, args.dir_pelvis, args.dir_output, anomaly_range)

    logger.info("✅ Dataset Creation Complete.")