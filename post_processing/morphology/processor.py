"""
Morphological Post-processing

Change: Filter small components BEFORE morphological operations

Data flow:
  masked pred_mask (0/1 or 0/255) → filter small components → dilate × N → erode × N → repeat → output (0/1)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image


@dataclass
class ProcessingStats:
    """Statistics for a single mask processing"""
    original_area: int
    after_filtering_area: int  # NEW: area after early filtering
    after_morphology_area: int
    final_area: int  # Final output area
    total_components_original: int  # NEW: components before filtering
    total_components_filtered: int  # NEW: components after filtering
    components_removed_early: int  # NEW: removed by early filtering
    area_preserved_ratio: float
    
    def to_dict(self) -> Dict:
        return {
            'original_area': self.original_area,
            'after_early_filtering': self.after_filtering_area,
            'after_morphology': self.after_morphology_area,
            'final_area': self.final_area,
            'components_original': self.total_components_original,
            'components_after_filtering': self.total_components_filtered,
            'components_removed_early': self.components_removed_early,
            'area_preserved_ratio': f"{self.area_preserved_ratio:.2%}",
        }


class MorphologyProcessor:
    """
    Morphological processing for binary masks
    
    NEW OPERATION SEQUENCE:
    1. Filter connected components < min_component_size (EARLY NOISE REMOVAL)
    2. Dilation × dilate_iterations (expand remaining anomalies)
    3. Erosion × erode_iterations (restore size, keep connected areas)
    4. Repeat num_rounds times
    """
    
    def __init__(
        self,
        dilate_iterations: int = 1,
        erode_iterations: int = 1,
        num_rounds: int = 1,
        kernel_size: int = 5,
        kernel_shape: str = "ellipse",
        min_component_size: int = 3,
        connectivity: int = 8,
    ):
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if kernel_size < 3:
            raise ValueError(f"kernel_size must be >= 3, got {kernel_size}")
        
        self.dilate_iterations = dilate_iterations
        self.erode_iterations = erode_iterations
        self.num_rounds = num_rounds
        self.kernel_size = kernel_size
        self.kernel_shape = kernel_shape
        self.min_component_size = min_component_size
        self.connectivity = connectivity
        
        # Create morphological kernel
        if kernel_shape == "ellipse":
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )
        else:
            self.kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, kernel_size)
            )
    
    def _binarize(self, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binarize mask to {0, 1}
        
        Handles both float [0,1] and uint8 [0,255] inputs
        """
        if mask.dtype == np.uint8:
            binary = (mask > (threshold * 255)).astype(np.uint8)
        else:
            binary = (mask > threshold).astype(np.uint8)
        return binary
    
    def _filter_small_components(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Remove connected components with area < min_component_size
        
        NEW: This is now called BEFORE morphological operations
        
        Returns:
            filtered_mask: uint8 binary mask {0, 1}
            stats: dict with component statistics
        """
        # Ensure uint8
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0).astype(np.uint8) * 255
        else:
            # Convert {0,1} to {0,255} for connectedComponentsWithStats
            binary_mask = mask * 255 if mask.max() <= 1 else mask
        
        # Find connected components
        num_labels, labels_map, stats_cv, centroids = cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=self.connectivity
        )
        
        # Build filtered mask
        filtered_mask = np.zeros_like(labels_map, dtype=np.uint8)
        kept = 0
        removed = 0
        
        for label in range(1, num_labels):  # 0 is background
            area = stats_cv[label, cv2.CC_STAT_AREA]
            
            if area >= self.min_component_size:
                filtered_mask[labels_map == label] = 1
                kept += 1
            else:
                removed += 1
        
        stats_dict = {
            'total_components': num_labels - 1,
            'kept_components': kept,
            'removed_components': removed,
        }
        
        return filtered_mask, stats_dict
    
    def _apply_morphological_ops(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing operation num_rounds times
        
        NEW: This is now called AFTER filtering small components
        
        Each round: dilation × dilate_iterations → erosion × erode_iterations
        """
        result = mask.copy()
        
        for round_idx in range(self.num_rounds):
            # Dilation: expand anomalies, connect nearby fragments
            for _ in range(self.dilate_iterations):
                result = cv2.dilate(result, self.kernel, iterations=1)
            
            # Erosion: restore approximate size, but keep connected regions
            for _ in range(self.erode_iterations):
                result = cv2.erode(result, self.kernel, iterations=1)
        
        return result
    
    def process(
        self,
        mask: np.ndarray,
        binarize_threshold: float = 0.5
    ) -> Tuple[np.ndarray, ProcessingStats]:
        """
        Process single mask through complete morphology pipeline
        
        NEW SEQUENCE:
        1. Binarize
        2. Filter small components (EARLY)
        3. Apply morphological operations
        
        Args:
            mask: Input mask (any range, will be binarized)
            binarize_threshold: Threshold for binarization
        
        Returns:
            processed_mask: Output mask (uint8, values 0 or 1)
            stats: Processing statistics
        """
        # Step 1: Binarize and record original area
        binary = self._binarize(mask, binarize_threshold)
        original_area = np.count_nonzero(binary)
        
        # Count original components
        num_labels_orig = cv2.connectedComponents(binary * 255, connectivity=self.connectivity)[0]
        total_components_original = num_labels_orig - 1  # Exclude background
        
        # Step 2: Filter small components FIRST (EARLY NOISE REMOVAL)
        filtered_early, cc_stats = self._filter_small_components(binary)
        filtered_area = np.count_nonzero(filtered_early)
        
        # Step 3: Apply morphological operations to filtered mask
        morpho_result = self._apply_morphological_ops(filtered_early)
        final_area = np.count_nonzero(morpho_result)
        
        # Step 4: Build statistics
        stats = ProcessingStats(
            original_area=original_area,
            after_filtering_area=filtered_area,
            after_morphology_area=final_area,
            final_area=final_area,  # Same as after_morphology_area in current version
            total_components_original=total_components_original,
            total_components_filtered=cc_stats['kept_components'],
            components_removed_early=cc_stats['removed_components'],
            area_preserved_ratio=final_area / max(original_area, 1),
        )
        
        return morpho_result, stats


class BatchProcessor:
    """Batch processing of multiple mask files"""
    
    def __init__(self, processor: MorphologyProcessor):
        self.processor = processor
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        binarize_threshold: float = 0.5,
        max_samples: int | None = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Process all masks in a directory
        
        Args:
            input_dir: Input directory containing masks
            output_dir: Output directory for processed masks
            binarize_threshold: Threshold for binarization
            max_samples: Limit processing to N samples (for testing)
            verbose: Print progress information
        
        Returns:
            Dictionary with processing statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect mask files
        mask_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.npy")))
        
        if max_samples is not None:
            mask_files = mask_files[:max_samples]
        
        if not mask_files:
            raise FileNotFoundError(f"No mask files found in {input_dir}")
        
        # Processing statistics
        total_stats = {
            'processed': 0,
            'failed': 0,
            'failed_files': [],
            'avg_area_preserved': 0,
            'total_components_removed_early': 0,  # NEW
            'individual_stats': []
        }
        
        for idx, mask_file in enumerate(mask_files):
            try:
                # Load mask
                if mask_file.suffix.lower() == '.npy':
                    mask = np.load(mask_file)
                else:
                    img = Image.open(mask_file).convert('L')
                    mask = np.array(img)
                
                # Process
                processed, stats = self.processor.process(
                    mask,
                    binarize_threshold=binarize_threshold
                )
                
                # Save output (keep same format as input)
                output_file = output_dir / mask_file.name
                if mask_file.suffix.lower() == '.npy':
                    np.save(output_file, processed)
                else:
                    # Save as PNG (0 or 1 as uint8, displayed as 0 or 255)
                    Image.fromarray(processed * 255).save(output_file)
                
                # Update statistics
                total_stats['processed'] += 1
                total_stats['avg_area_preserved'] += stats.area_preserved_ratio
                total_stats['total_components_removed_early'] += stats.components_removed_early
                total_stats['individual_stats'].append({
                    'file': mask_file.name,
                    'stats': stats.to_dict()
                })
                
                if verbose and (idx + 1) % 10 == 0:
                    print(f"[INFO] Processed {idx + 1}/{len(mask_files)} files")
                
            except Exception as e:
                total_stats['failed'] += 1
                total_stats['failed_files'].append({
                    'file': mask_file.name,
                    'error': str(e)
                })
                if verbose:
                    print(f"[WARN] Failed to process {mask_file.name}: {e}")
        
        # Compute average
        if total_stats['processed'] > 0:
            total_stats['avg_area_preserved'] /= total_stats['processed']
        
        if verbose:
            self._print_summary(total_stats, len(mask_files))
        
        return total_stats
    
    @staticmethod
    def _print_summary(total_stats: Dict, total_files: int) -> None:
        """Print processing summary"""
        print("\n" + "="*70)
        print("Processing Summary")
        print("="*70)
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {total_stats['processed']}")
        print(f"Failed: {total_stats['failed']}")
        print(f"Average area preserved: {total_stats['avg_area_preserved']:.2%}")
        print(f"Total components removed (early filtering): {total_stats['total_components_removed_early']}")
        print("="*70 + "\n")