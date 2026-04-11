import os
from pathlib import Path
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
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Center PNG pelvis preprocessing")
    parser.add_argument("--dir_pelvis", type=str, required=True)
    parser.add_argument("--dir_output", type=str, required=True)
    return parser.parse_args()

# ================================================================
# CORE PROCESSING
# ================================================================
def process_slices(
    mr_norm, body_mask_vol, id_, split, subset,
    output_dir, start_idx=SLICE_INDEX_START_NORMAL, end_offset=SLICE_INDEX_END_NORMAL,
    mask_vol=None, abnormal_slices=None
):
    slices = mr_norm.shape[2]
    slice_indices = (
        range(start_idx, slices + end_offset)
        if abnormal_slices is None else abnormal_slices
    )

    for i in tqdm(slice_indices, desc=f"{id_}-{split}-{subset}"):
        slice_img = mr_norm[:, :, i]
        slice_body_mask = body_mask_vol[:, :, i]
        slice_mask = mask_vol[:, :, i] if mask_vol is not None else np.zeros_like(slice_body_mask)

        # Padding, Resizing, and Cropping logic
        s_img_p, (ph, pw) = center_pad_single_slice(slice_img)
        s_bm_p = center_pad_single_slice_by_params(slice_body_mask, ph, pw)
        s_m_p = center_pad_single_slice_by_params(slice_mask, ph, pw)

        img_crop = center_crop(resize_image(s_img_p, target_size=TARGET_SIZE))
        bm_crop = center_crop(resize_image(s_bm_p, target_size=TARGET_SIZE))
        mask_crop = center_crop(resize_image(s_m_p, target_size=TARGET_SIZE))

        if mask_vol is not None and mask_crop.sum() < 3:
            continue

        if split == "train":
            save_png(img_crop, os.path.join(output_dir, split, subset, f"{id_}_{i}.png"), cmap="bone")
            save_png(bm_crop, os.path.join(output_dir, split, "bodymask", f"{id_}_{i}.png"), cmap="binary")
        else:
            base_path = os.path.join(output_dir, split, subset)
            save_png(img_crop, os.path.join(base_path, "img", f"{id_}_{i}.png"), cmap="bone")
            save_png(mask_crop, os.path.join(base_path, "label", f"{id_}_{i}.png"), cmap="binary")
            save_png(bm_crop, os.path.join(base_path, "bodymask", f"{id_}_{i}.png"), cmap="binary")

# ================================================================
# SPLIT HANDLERS
# ================================================================
def process_good_scans(det, ids, split, dir_pelvis, dir_output):
    for id_ in ids:
        _, mr_norm, _, body_mask_vol = load_scan(dir_pelvis, det, id_)
        process_slices(mr_norm, body_mask_vol, id_, split, "good", dir_output)

def process_ungood_scans(det, ids, split, dir_pelvis, dir_output, anomaly_range):
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_)
        slices = mr_norm.shape[2]
        abnormal_slices = list(range(anomaly_range[id_][0], anomaly_range[id_][-1]))

        # Thresholding and Refinement logic
        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], abnormal_slices).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        
        tau = min(scan_value - DELTA, 2000)
        mask_vol = (ct >= tau).astype(np.uint8)
        mask_ref = det.refine_mask_with_mr(mask_vol, mr, lo_diff=5, up_diff=10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, min_area_for_smooth=50, slice_axis=2)

        safe_abnormal_slices = [i for i in abnormal_slices if 15 <= i < slices - 15]
        process_slices(mr_norm, body_mask_vol, id_, split, "Ungood", dir_output, 
                       mask_vol=mask_ref, abnormal_slices=safe_abnormal_slices)

# ================================================================
# PATIENT-WISE EXPORT
# ================================================================
def export_full_anomalous_cases_png(det, ids, dir_pelvis, dir_output, anomaly_range):
    for id_ in ids:
        mr, mr_norm, ct, body_mask_vol = load_scan(dir_pelvis, det, id_)
        slices = mr_norm.shape[2]
        start_idx, end_idx = 10, slices - 5
        if end_idx <= start_idx: continue

        # Redo refinement for the whole patient range
        abn_range = anomaly_range.get(id_, (start_idx, end_idx))
        df_hu = det.score_volume_hu(ct, scan_id=id_, slice_axis=2)
        df_hu["label"] = np.isin(df_hu["slice_idx"], range(*abn_range)).astype(np.uint8)
        scan_value, _ = det.pick_global_tau_by_hu(df_hu, label_col="label")
        
        tau = min(scan_value - DELTA, 2000)
        mask_ref = det.refine_mask_with_mr((ct >= tau).astype(np.uint8), mr, lo_diff=5, up_diff=10)
        mask_ref = det.postprocess_mask_volume_morph(mask_ref, disk_size=5, slice_axis=2)

        base_path = os.path.join(dir_output, "test", "Ungood_whole_patient_scans")
        
        for i in tqdm(range(start_idx, end_idx), desc=f"Patient-wise {id_}"):
            s_img_p, (ph, pw) = center_pad_single_slice(mr_norm[:, :, i])
            s_bm_p = center_pad_single_slice_by_params(body_mask_vol[:, :, i], ph, pw)
            s_m_p = center_pad_single_slice_by_params(mask_ref[:, :, i], ph, pw)

            img_crop = center_crop(resize_image(s_img_p, target_size=TARGET_SIZE))
            bm_crop = center_crop(resize_image(s_bm_p, target_size=TARGET_SIZE))
            m_crop = center_crop(resize_image(s_m_p, target_size=TARGET_SIZE))

            save_png(img_crop, os.path.join(base_path, "img", f"{id_}_{i}.png"), cmap="bone")
            save_png(m_crop, os.path.join(base_path, "label", f"{id_}_{i}.png"), cmap="binary")
            save_png(bm_crop, os.path.join(base_path, "bodymask", f"{id_}_{i}.png"), cmap="binary")

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    create_output_dirs(args.dir_output)

    # 1. Load Data Splits
    df_overview = pd.read_excel(os.path.join(args.dir_pelvis, "overview/1_pelvis_train.xlsx"), sheet_name="MR")
    ids_all = [i for i in df_overview["ID"].tolist() if i.startswith("1P")]

    root_dir = Path(__file__).resolve().parents[2] 
    label_implant_path = root_dir / "labels" / "labels_implant.json"
    label_others_path = root_dir / "labels" / "labels_others.json"

    with open(label_implant_path, "r") as f:
        raw_data = json.load(f)['type1']

    implant_list = raw_data['type1'] if isinstance(raw_data, dict) and 'type1' in raw_data else raw_data
    df_labels_1 = pd.DataFrame([{"id": k, **v} for item in implant_list for k, v in item.items()])
    list_na_ids = df_labels_1[df_labels_1.isna().any(axis=1)]['id'].tolist()
    df_labels_1 = df_labels_1.dropna().query("body_part == 'pelvis'")
    
    anomaly_range = {row.id: (int(min(row.ct_start, row.mr_start)), int(max(row.ct_end, row.mr_end))) 
                     for row in df_labels_1.itertuples()}

    ids_abnormal_all = df_labels_1['id'].tolist()
    # MC Specific Exclusions
    for exclude in ['1PA030', '1PA170', '1PC029', '1PC015']:
        if exclude in ids_abnormal_all: ids_abnormal_all.remove(exclude)

    with open(label_others_path, "r") as f:
        data_other = json.load(f)
    ids_other = [pid for item in data_other['types_2_to_7'] for pid, info in item.items() 
                 if pid.startswith("1P") and str(info.get("type")) in {"2", "3", "4", "5", "6"}]

    ids_used = list(set(ids_all) - set(list_na_ids))
    ids_normal_all = sorted(set(ids_used) - set(ids_abnormal_all) - set(ids_other))

    # 2. Split Logic
    random.shuffle(ids_normal_all)
    random.shuffle(ids_abnormal_all)

    v_frac, t_frac = 0.1, 0.2
    n_val_abn = max(4, int(len(ids_abnormal_all) * v_frac))
    n_test_abn = max(1, int(len(ids_abnormal_all) * t_frac))
    ids_abn_valid = ids_abnormal_all[:n_val_abn]
    ids_abn_test = ids_abnormal_all[n_val_abn:]

    n_val_norm = max(1, int(len(ids_normal_all) * v_frac))
    n_test_norm = max(1, int(len(ids_normal_all) * t_frac))
    ids_norm_valid = ids_normal_all[:n_val_norm]
    ids_norm_test = ids_normal_all[n_val_norm:n_val_norm + n_test_norm]
    ids_norm_train = ids_normal_all[n_val_norm + n_test_norm:]

    # 3. Execution
    det = MetalArtifactDetector()
    
    logger.info("Processing Splits...")
    process_good_scans(det, ids_norm_train, "train", args.dir_pelvis, args.dir_output)
    process_good_scans(det, ids_norm_val, "valid", args.dir_pelvis, args.dir_output)
    process_ungood_scans(det, ids_abn_val, "valid", args.dir_pelvis, args.dir_output, anomaly_range)
    process_good_scans(det, ids_norm_test, "test", args.dir_pelvis, args.dir_output)
    process_ungood_scans(det, ids_abn_test, "test", args.dir_pelvis, args.dir_output, anomaly_range)

    # 4. Patient-wise Export
    ids_eff = sorted(get_ids_from_ungood_test_folder(args.dir_output).intersection(ids_abn_test))
    export_full_anomalous_cases_png(det, ids_eff, args.dir_pelvis, args.dir_output, anomaly_range)

    logger.info("✅ Multi-Center PNG processing complete.")