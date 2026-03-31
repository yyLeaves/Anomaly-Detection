import os
import numpy as np
import cv2
from .io_utils import load_nifti_image

def apply_mask(mr_image, mask):
    return mr_image * (mask > 0).astype(mr_image.dtype)

def center_pad_single_slice(image):
    h, w = image.shape
    max_size = max(h, w)
    pad_h, pad_w = (max_size - h) // 2, (max_size - w) // 2
    square_slice = np.zeros((max_size, max_size), dtype=image.dtype)
    square_slice[pad_h:pad_h+h, pad_w:pad_w+w] = image
    return square_slice, (pad_h, pad_w)

def center_pad_single_slice_by_params(image, pad_h, pad_w):
    h, w = image.shape
    max_size = max(h, w)
    square_slice = np.zeros((max_size, max_size), dtype=image.dtype)
    square_slice[pad_h:pad_h+h, pad_w:pad_w+w] = image
    return square_slice

def resize_image(image, target_size=(240, 240)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST_EXACT)

def center_crop(slice_, target_size=(224, 224)):
    h, w = slice_.shape
    th, tw = target_size
    i, j = int(round((h - th) / 2.0)), int(round((w - tw) / 2.0))
    return slice_[i:i + th, j:j + tw]

def minmax_normalize_numpy(volume, clip_range=(0, 2000)):
    v = volume.astype(np.float32).clip(*clip_range)
    v_min, v_max = np.min(v), np.max(v)
    if v_max > v_min:
        v = (v - v_min) / (v_max - v_min) * 255
    else:
        v = np.zeros_like(v)
    return v.astype(np.uint8)

def extract_3ch_slice_rep(volume, i):
    """Extract a 3-channel slice (current, current, current)."""
    idxs = [i, i, i]
    return volume[:, :, idxs].copy()

def extract_3ch_slice_con(volume, i, slice_index_start, slice_index_end):
    """
    Extract a 3-channel slice with temporal context (previous, current, next).
    Handles boundaries by repeating the current slice if neighbors are out of bounds.
    """
    slices = volume.shape[2]
    
    # Boundary check for the very first slice in the range
    if i == slice_index_start:
        idxs = [i, i, i + 1]
    # Boundary check for the very last slice in the range
    elif i == slices + slice_index_end - 1:
        idxs = [i - 1, i, i]
    else:
        idxs = [i - 1, i, i + 1]
        
    return volume[:, :, idxs].copy()

def load_scan(dir_pelvis, det, id_, thresh_mr_mask=0.1):
    """Logic to load and prepare a volume."""
    dir_scan = os.path.join(dir_pelvis, id_)
    mr = load_nifti_image(os.path.join(dir_scan, "mr.nii.gz"))
    ct = load_nifti_image(os.path.join(dir_scan, "ct.nii.gz"))
    mask = load_nifti_image(os.path.join(dir_scan, "mask.nii.gz"))

    body_mask_vol = det.get_body_mask_threshold(mr * mask, threshold_ct_body_mask=thresh_mr_mask)
    body_mask_vol = np.logical_and(body_mask_vol > 0, mask > 0).astype(np.uint8)
    
    mr_norm = minmax_normalize_numpy(apply_mask(mr, body_mask_vol))
    return mr, mr_norm, ct, body_mask_vol