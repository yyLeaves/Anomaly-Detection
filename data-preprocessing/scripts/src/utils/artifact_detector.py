import numpy as np
import pandas as pd
import cv2

from skimage import morphology, measure
from scipy.ndimage import binary_erosion, binary_dilation



class MetalArtifactDetector:
    def __init__(self, metric='f1'):
        """
        metric: ('f1', 'youden', 'balanced_acc', 'recall', 'precision', 'accuracy')
        """
        self.metric = metric
        self.tau_global = None
        self.tau_map = {}

    @staticmethod
    def _confusion_binary(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.uint8)
        y_pred = np.asarray(y_pred, dtype=np.uint8)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return tp, fp, fn, tn

    @staticmethod
    def _metrics_from_cm(tp, fp, fn, tn, eps=1e-9):
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        tpr  = rec
        tnr  = tn / (tn + fp + eps)
        bal_acc = 0.5 * (tpr + tnr)
        youden  = tpr + tnr - 1.0
        acc     = (tp + tn) / (tp + fp + fn + tn + eps)
        return {
            "precision": prec, "recall": rec, "f1": f1,
            "balanced_acc": bal_acc, "youden": youden, "accuracy": acc
        }

    @staticmethod
    def _norm01(x, hu_window=None, clip_quantiles=None, eps=1e-6):
        x = np.asarray(x, dtype=float)
        if hu_window is not None:
            lo, hi = hu_window
            x = np.clip(x, lo, hi)
        elif clip_quantiles is not None:
            qlo, qhi = clip_quantiles
            lo = np.percentile(x, qlo*100.0)
            hi = np.percentile(x, qhi*100.0)
            x = np.clip(x, lo, hi)
        mn, mx = x.min(), x.max()
        if mx - mn < eps:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn + eps)

    def score_volume_hu(self, vol, scan_id="scan1", slice_axis=0):
        """return max HU for each slice"""
        vol_z = np.moveaxis(vol, slice_axis, 0)  # (Z,H,W)
        z, h, w = vol_z.shape
        slice_max = np.max(vol_z.reshape(z, -1), axis=1)
        return pd.DataFrame({
            "scan_id": scan_id,
            "slice_idx": np.arange(z),
            "slice_max_hu": slice_max.astype(float),
        })

    def pick_global_tau_by_hu(self, df, metal_slices=None,
                              label_col='label', hu_col='slice_max_hu'):
        """
        global_tau = max(max_normal_hu, min_max_abnormal_hu)
        """
        if metal_slices is None:
            if label_col not in df.columns:
                raise ValueError("Need metal_slices or df[label_col].")
            labels = df[label_col].values.astype(int)
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
        else:
            sl = df["slice_idx"].values
            pos_mask = np.isin(sl, np.asarray(metal_slices, dtype=int))
            pos_idx = np.where(pos_mask)[0]
            neg_idx = np.where(~pos_mask)[0]

        vals = df[hu_col].values.astype(float)

        max_normal_hu = float(vals[neg_idx].max()) if len(neg_idx) > 0 else -np.inf
        min_max_abnormal_hu = float(vals[pos_idx].min()) if len(pos_idx) > 0 else np.inf

        global_tau = max(max_normal_hu, min_max_abnormal_hu)
        self.tau_global = global_tau
        return global_tau, {
            "max_normal_hu": max_normal_hu,
            "min_max_abnormal_hu": min_max_abnormal_hu
        }

    def apply_tau_by_hu(self, df, hu_col='slice_max_hu'):
        """classify slices based on HU values"""
        if self.tau_global is None or not np.isfinite(self.tau_global):
            raise ValueError("Global tau not set. Run pick_global_tau_by_hu first.")
        out = df.copy()
        out["pred"] = (out[hu_col].values >= self.tau_global).astype(np.uint8)
        return out

    def evaluate(self, df, label_col='label', pred_col='pred', scan_col='scan_id'):
        # slice-level
        tp, fp, fn, tn = self._confusion_binary(df[label_col].values, df[pred_col].values)
        sl = self._metrics_from_cm(tp, fp, fn, tn)
        sl.update(dict(tp=tp, fp=fp, fn=fn, tn=tn))

        # scan-level
        scan_true = df.groupby(scan_col, sort=False)[label_col].max().values
        scan_pred = df.groupby(scan_col, sort=False)[pred_col].max().values
        tp, fp, fn, tn = self._confusion_binary(scan_true, scan_pred)
        sc = self._metrics_from_cm(tp, fp, fn, tn)
        sc.update(dict(tp=tp, fp=fp, fn=fn, tn=tn))
        return {"slice_level": sl, "scan_level": sc}


    def extract_mask_volume(self, vol, tau, slice_axis=0):
        return (vol >= tau).astype(np.uint8)


    def get_mask_biggest_contour(self, mask_ct):
        for i in range(mask_ct.shape[2]):
            inmask = np.expand_dims(mask_ct[:, :, i].astype(np.uint8), axis=2)
            ret, bin_img = cv2.threshold(inmask, 0.5, 1, cv2.THRESH_BINARY)
            (cnts, _) = cv2.findContours(np.expand_dims(bin_img, axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # return None, if no contours detected
            if len(cnts) != 0:
                # based on contour area, get the maximum contour which is a body contour
                segmented = max(cnts, key=cv2.contourArea)
                bin_img[bin_img > 0] = 0
                a = cv2.drawContours(np.expand_dims(bin_img, axis=2), [segmented], 0, (255, 255, 255), -1)
                a[a > 0] = 1
                mask_ct[:, :, i] = a.squeeze()
        return mask_ct.astype(np.uint8)

    def get_body_mask_threshold(self, nii_array, threshold_ct_body_mask):
        mask_ct = np.zeros(nii_array.shape)
        mask_ct[nii_array > threshold_ct_body_mask] = 1
        mask_ct[nii_array <= threshold_ct_body_mask] = 0
        mask_ct = binary_erosion(mask_ct, iterations=2).astype(np.uint8)
        mask_ct = self.get_mask_biggest_contour(mask_ct)
        mask_ct = binary_dilation(mask_ct, iterations=5).astype(np.int16)
        return mask_ct
    
    
    def refine_mask_with_mr(self, ct_mask_vol, mr_vol, lo_diff=10, up_diff=150, min_contour_area=5):
        """
        Refines a 3D CT mask volume using a corresponding MR volume by performing
        a flood-fill operation on each slice. Handles small, single-pixel anomalies
        by preserving the original CT mask.
        """
        assert ct_mask_vol.shape == mr_vol.shape, "CT mask and MR must have the same shape"
        
        # Create an empty volume to store the refined masks
        refined_mask = np.zeros_like(ct_mask_vol, dtype=np.uint8)

        # Iterate over each slice in the 3D volumes
        for i in range(ct_mask_vol.shape[2]):
            slice_ct = ct_mask_vol[:, :, i].astype(np.uint8)
            slice_mr = mr_vol[:, :, i]

            # If there's no mask on this slice, skip it
            if slice_ct.sum() == 0:
                continue

            # Find all contours on this slice
            contours, _ = cv2.findContours(slice_ct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue

            # Store the combined result of all flood fills and preserved masks
            refined_mask_slice = np.zeros_like(slice_ct)

            for cnt in contours:
                contour_area = cv2.contourArea(cnt)
                
                # Check if the contour is too small
                if contour_area < min_contour_area:
                    # If too small, keep the original mask 
                    mask_cnt = np.zeros(slice_ct.shape, dtype=np.uint8)
                    cv2.drawContours(mask_cnt, [cnt], -1, 1, -1)
                    refined_mask_slice = np.logical_or(refined_mask_slice, mask_cnt).astype(np.uint8)
                    continue

                # Calculate the centroid (x, y) coordinates 
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Perform the flood-fill operation 
                flood_fill_slice = slice_mr.copy().astype(np.float32)

                # Create an empty mask to store the result of the flood fill.
                h, w = flood_fill_slice.shape
                mask = np.zeros((h + 2, w + 2), np.uint8)

                cv2.floodFill(flood_fill_slice, mask, (cx, cy), 255, lo_diff, up_diff, cv2.FLOODFILL_MASK_ONLY)

                # Remove the extra border from the mask to get the final refined mask for the slice.
                current_fill = mask[1:-1, 1:-1]
                
                # Combine the new flood fill result with the existing refined mask for the slice
                refined_mask_slice = np.logical_or(refined_mask_slice, current_fill).astype(np.uint8)

            refined_mask[:, :, i] = refined_mask_slice

        return refined_mask

    def postprocess_mask_morph(self, mask2d, disk_size=3, min_area_for_smooth=50):
        """
        Morphological postprocessing for binary masks with conditional smoothing
        based on anomaly area.
        """
        mask = mask2d.astype(np.uint8)

        # Compute total anomaly area in this slice
        total_area = np.count_nonzero(mask)

        # Only apply smoothing if the total area exceeds threshold
        if total_area >= min_area_for_smooth:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disk_size, disk_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask.astype(np.uint8)

    def postprocess_mask_volume_morph(self, mask_vol, slice_axis=2, disk_size=3, min_area_for_smooth=50):
        """
        Apply conditional morphological postprocessing slice-by-slice.
        """
        vol = np.moveaxis(mask_vol, slice_axis, 0)
        out_slices = []
        for z in range(vol.shape[0]):
            out_slices.append(self.postprocess_mask_morph(
                vol[z],
                disk_size=disk_size,
                min_area_for_smooth=min_area_for_smooth
            ))
        out_vol = np.stack(out_slices, axis=0)
        return np.moveaxis(out_vol, 0, slice_axis)