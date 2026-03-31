import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
import cv2


def load_nifti_image(image_path):
    img = nib.load(image_path)
    return img.get_fdata()


def apply_mask(mr_image, mask):
    if mr_image.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {mr_image.shape} vs {mask.shape}")
    return mr_image * (mask > 0).astype(mr_image.dtype)


def center_pad_single_slice(image):
    h, w = image.shape
    max_size = max(h, w)
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    square_slice = np.zeros((max_size, max_size), dtype=image.dtype)
    square_slice[pad_h:pad_h + h, pad_w:pad_w + w] = image
    return square_slice, (pad_h, pad_w)


def minmax_normalize_numpy(volume, clip_range=(0, 2000)):
    v = volume.astype(np.float32)
    v = v.clip(*clip_range)
    v_min, v_max = np.min(v), np.max(v)
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min) * 255
    else:
        v = np.zeros_like(v)
    return v.astype(np.uint8)


def get_mask_biggest_contour(mask_ct):
    for i in range(mask_ct.shape[2]):
        slice_2d = mask_ct[:, :, i].astype(np.uint8)
        _, bin_img = cv2.threshold(slice_2d, 0.5, 1, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) != 0:
            segmented = max(cnts, key=cv2.contourArea)
            out_mask = np.zeros_like(bin_img)
            cv2.drawContours(out_mask, [segmented], 0, 1, -1)
            mask_ct[:, :, i] = out_mask
    return mask_ct.astype(np.uint8)


def get_body_mask_threshold(nii_array, threshold_ct_body_mask):
    mask_ct = np.zeros(nii_array.shape)
    mask_ct[nii_array > threshold_ct_body_mask] = 1
    mask_ct = binary_erosion(mask_ct, iterations=2).astype(np.uint8)
    mask_ct = get_mask_biggest_contour(mask_ct)
    mask_ct = binary_dilation(mask_ct, iterations=5).astype(np.int16)
    return mask_ct


def preprocess_mri_to_224_cropped(mr_image, remove_first=10, remove_last=5, thresh_body=0.1, target_size=(224, 224)):
    if remove_last == 0:
        mr_cropped = mr_image[:, :, remove_first:]
    else:
        mr_cropped = mr_image[:, :, remove_first:-remove_last]
    body_mask = get_body_mask_threshold(mr_cropped, threshold_ct_body_mask=thresh_body)
    body_mask = np.logical_and(body_mask > 0, np.ones_like(mr_cropped, dtype=bool))
    masked_mr = apply_mask(mr_cropped, body_mask)
    mr_norm = minmax_normalize_numpy(masked_mr)
    h, w, z = mr_norm.shape
    max_size = max(h, w)
    vol_square = np.zeros((max_size, max_size, z), dtype=mr_norm.dtype)
    for k in range(z):
        square_slice, _ = center_pad_single_slice(mr_norm[:, :, k])
        vol_square[:, :, k] = square_slice
    vol_resized = np.zeros((target_size[1], target_size[0], z), dtype=mr_norm.dtype)
    for k in range(z):
        vol_resized[:, :, k] = cv2.resize(vol_square[:, :, k], target_size, interpolation=cv2.INTER_AREA)
    return vol_resized


def create_overlay(base_img, mask):
    overlay = np.zeros(base_img.shape + (4,), dtype=np.float32)
    overlay[mask > 0] = [1.0, 0.0, 0.0, 0.5]  # Red overlay
    return overlay


def plot_slice(mr_slice_r, gt_slice_r, pred_slice_r, slice_idx, output_dir, fmt, model_name="Prediction"):
    # Updated to 1x3 layout
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for ax in axs:
        ax.axis('off')

    # 1. Clean MR
    axs[0].imshow(mr_slice_r, cmap='gray')
    axs[0].set_title('MR', fontsize=14)

    # 2. Ground Truth Overlay
    axs[1].imshow(mr_slice_r, cmap='gray')
    axs[1].imshow(create_overlay(mr_slice_r, gt_slice_r))
    axs[1].set_title('Ground Truth', fontsize=14)

    # 3. Prediction Overlay
    axs[2].imshow(mr_slice_r, cmap='gray')
    axs[2].imshow(create_overlay(mr_slice_r, pred_slice_r))
    axs[2].set_title(model_name, fontsize=14)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.05, wspace=0.02)

    fname = f'slice_{slice_idx:03d}.{fmt}'
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MR anomaly visualization')
    parser.add_argument('--mr_path', required=True)
    parser.add_argument('--predicted_mask_path', required=True)
    parser.add_argument('--ground_truth_path', required=True)
    parser.add_argument('--output_directory', required=True)
    parser.add_argument('--saving_format', choices=['png', 'pdf'], default='png')
    parser.add_argument('--patient_slice', type=int)
    parser.add_argument('--model_name', default='RD4AD')
    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    print("Loading volumes...")
    mri_raw = load_nifti_image(args.mr_path)
    gt_raw = load_nifti_image(args.ground_truth_path)
    gt_bin = (gt_raw > 0).astype(np.uint8)

    pred_raw = load_nifti_image(args.predicted_mask_path)
    pred_bin = (pred_raw > 0).astype(np.uint8)

    print("Preprocessing MR...")
    mr_resized = preprocess_mri_to_224_cropped(mri_raw)

    if args.patient_slice is not None:
        slices = [args.patient_slice]
    else:
        combined_masks = gt_bin + pred_bin
        slices = np.where(np.any(combined_masks, axis=(0, 1)))[0]

    print(f"Processing {len(slices)} slices")

    for slice_idx in slices:
        mr_slice_full = mr_resized[:, :, slice_idx]

        thr, margin = 5, 5
        mask_nonzero = mr_slice_full > thr
        if not np.any(mask_nonzero): continue

        cols = np.any(mask_nonzero, axis=0)
        col_min, col_max = np.where(cols)[0][[0, -1]]
        col_min = max(col_min - margin, 0)
        col_max = min(col_max + margin, mr_slice_full.shape[1] - 1)

        mr_c = mr_resized[:, col_min:col_max + 1, slice_idx]
        gt_c = gt_bin[:, col_min:col_max + 1, slice_idx]
        pred_c = pred_bin[:, col_min:col_max + 1, slice_idx]

        left_crop, right_crop = 5, 5
        mr_r = np.rot90(mr_c, k=-1)[:, left_crop:-right_crop]
        gt_r = np.rot90(gt_c, k=-1)[:, left_crop:-right_crop]
        pred_r = np.rot90(pred_c, k=-1)[:, left_crop:-right_crop]

        plot_slice(mr_r, gt_r, pred_r, slice_idx, args.output_directory, args.saving_format, args.model_name)
        print(f"Saved slice {slice_idx}")


if __name__ == '__main__':
    main()