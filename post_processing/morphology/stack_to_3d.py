#!/usr/bin/env python3
"""
Batch Stack Entire Directory Tree to Patient-wise NIfTI Volumes

After applying morphology to entire model directory tree, use this script to
convert all processed PNG masks to 3D NIfTI volumes while preserving directory structure.

Usage Examples:
    # Stack aggressive results
    python batch_stack_to_nifti.py \
      --model fastflow \
      --experiment aggressive \
      --spacing 1.0 1.0 1.0
    
    # Custom base directories
    python batch_stack_to_nifti.py \
      --model fastflow \
      --experiment aggressive \
      --input-base ./masks_morpho \
      --output-base ./nifti_3d
    

Directory structure:
    Input:  {input_base}/{model}_{experiment}/valid/Ungood/*.png
            {input_base}/{model}_{experiment}/test/Ungood/*.png
           
    Output: {output_base}/{model}_{experiment}/valid/Ungood/PA133.nii.gz
            {output_base}/{model}_{experiment}/test/Ungood/PA134.nii.gz
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm


class PatientGrouper:
    """Group PNG slices by patient ID"""
    
    def __init__(self, patient_pattern: str = r"PA(\d+)_"):
        self.patient_pattern = re.compile(patient_pattern)
    
    def extract_patient_id(self, filename: str) -> str | None:
        match = self.patient_pattern.search(filename)
        return match.group(1) if match else None
    
    def extract_slice_number(self, filename: str) -> int | None:
        match = re.search(r"_(\d+)\.", filename)
        return int(match.group(1)) if match else None
    
    def group_files(self, file_paths: List[Path]) -> Dict[str, List[Tuple[int, Path]]]:
        grouped = defaultdict(list)
        skipped = []
        
        for file_path in file_paths:
            patient_id = self.extract_patient_id(file_path.name)
            slice_num = self.extract_slice_number(file_path.name)
            
            if patient_id is None:
                skipped.append(file_path.name)
                continue
            
            if slice_num is None:
                slice_num = 0
            
            grouped[patient_id].append((slice_num, file_path))
        
        # Sort each patient's slices
        for patient_id in grouped:
            grouped[patient_id].sort(key=lambda x: x[0])
        
        return dict(grouped), skipped


class NIfTIStacker:
    """Convert 2D PNG slices to 3D NIfTI volume"""
    
    def __init__(self, 
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.spacing = spacing
        self.origin = origin
    
    def load_slices(self, slice_paths: List[Path]) -> np.ndarray:
        masks = []
        for png_path in slice_paths:
            img = Image.open(png_path).convert('L')
            mask = np.array(img)
            masks.append(mask)
        
        # Verify consistent dimensions
        shapes = [m.shape for m in masks]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent slice dimensions: {set(shapes)}")
        
        volume = np.stack(masks, axis=-1)
        volume = (volume > 127).astype(np.uint8)
        
        return volume
    
    def create_affine_matrix(self) -> np.ndarray:
        affine = np.eye(4)
        affine[0, 0] = self.spacing[0]
        affine[1, 1] = self.spacing[1]
        affine[2, 2] = self.spacing[2]
        affine[0, 3] = self.origin[0]
        affine[1, 3] = self.origin[1]
        affine[2, 3] = self.origin[2]
        return affine
    
    def save_nifti(self, volume: np.ndarray, output_path: Path) -> None:
        affine = self.create_affine_matrix()
        nifti_img = nib.Nifti1Image(volume, affine)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nifti_img, str(output_path))
    
    def stack_patient(self, slice_paths: List[Path], output_path: Path) -> Dict:
        volume = self.load_slices(slice_paths)
        self.save_nifti(volume, output_path)
        
        return {
            'num_slices': len(slice_paths),
            'shape': volume.shape,
            'non_zero_voxels': int(np.count_nonzero(volume)),
            'occupancy_percent': float(np.count_nonzero(volume) / volume.size * 100),
        }


class BatchNIfTIStacker:
    """Stack entire directory tree to NIfTI volumes"""
    
    def __init__(self,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 patient_pattern: str = r"PA(\d+)_"):
        self.stacker = NIfTIStacker(spacing, origin)
        self.grouper = PatientGrouper(patient_pattern)
        self.spacing = spacing
        self.patient_pattern = patient_pattern
    
    def find_mask_directories(self, base_dir: Path) -> List[Path]:
        """Find all directories containing PNG files"""
        mask_dirs = []
        for path in base_dir.rglob("*"):
            if path.is_dir() and list(path.glob("*.png")):
                mask_dirs.append(path)
        return sorted(mask_dirs)
    
    def get_relative_path(self, full_path: Path, base_dir: Path) -> Path:
        """Get relative path from base directory"""
        try:
            return full_path.relative_to(base_dir)
        except ValueError:
            return full_path
    
    def process_directory(self, 
                         input_dir: Path, 
                         output_dir: Path,
                         verbose: bool = False) -> Dict:
        """Process one directory (group by patient and stack)"""
        png_files = sorted(input_dir.glob("*.png"))
        
        if not png_files:
            return {'patients': 0, 'slices': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        
        # Group by patient
        grouped, skipped = self.grouper.group_files(png_files)
        
        if not grouped:
            if verbose:
                print(f"  [WARN] No patient IDs found in filenames")
            return {
                'patients': 0, 
                'slices': len(png_files), 
                'success': 0, 
                'failed': 0,
                'skipped': len(skipped)
            }
        
        stats = {
            'patients': len(grouped),
            'slices': len(png_files),
            'success': 0,
            'failed': 0,
            'skipped': len(skipped),
            'patient_results': []
        }
        
        # Process each patient
        for patient_id in sorted(grouped.keys()):
            slice_info = grouped[patient_id]
            slice_paths = [path for _, path in slice_info]
            output_path = output_dir / f"PA{patient_id}.nii.gz"
            
            try:
                patient_stats = self.stacker.stack_patient(slice_paths, output_path)
                stats['success'] += 1
                stats['patient_results'].append({
                    'patient_id': patient_id,
                    'output_file': str(output_path),
                    **patient_stats
                })
                
            except Exception as e:
                stats['failed'] += 1
                if verbose:
                    print(f"  [ERROR] Patient PA{patient_id}: {e}")
        
        return stats
    
    def process_tree(self,
                    input_base: Path,
                    output_base: Path,
                    dry_run: bool = False,
                    verbose: bool = True) -> Dict:
        """Process entire directory tree"""
        print("\n" + "="*70)
        print("Batch NIfTI Stacking (Patient-wise)")
        print("="*70)
        print(f"Input:  {input_base}")
        print(f"Output: {output_base}")
        print(f"Spacing: {self.spacing} mm")
        print(f"Patient pattern: {self.patient_pattern}")
        print("="*70 + "\n")
        
        # Find all directories with masks
        mask_dirs = self.find_mask_directories(input_base)
        
        if not mask_dirs:
            print(f"[ERROR] No directories with PNG files found in {input_base}")
            return {'total_dirs': 0, 'total_patients': 0, 'total_volumes': 0}
        
        print(f"[INFO] Found {len(mask_dirs)} directories with PNG files:\n")
        for dir_path in mask_dirs:
            rel_path = self.get_relative_path(dir_path, input_base)
            num_files = len(list(dir_path.glob("*.png")))
            print(f"  - {rel_path}/ ({num_files} files)")
        
        if dry_run:
            print("\n[DRY RUN] No processing performed. Remove --dry-run to process.")
            return {'total_dirs': len(mask_dirs), 'total_patients': 0, 'total_volumes': 0}
        
        print("\n" + "="*70)
        print("Processing...")
        print("="*70)
        
        # Process each directory
        summary = {
            'total_dirs': len(mask_dirs),
            'total_patients': 0,
            'total_volumes_created': 0,
            'total_volumes_failed': 0,
            'total_files_skipped': 0,
            'directory_results': []
        }
        
        for dir_path in tqdm(mask_dirs, desc="Processing directories"):
            rel_path = self.get_relative_path(dir_path, input_base)
            output_dir = output_base / rel_path
            
            if verbose:
                print(f"\n[PROCESSING] {rel_path}/")
            
            try:
                stats = self.process_directory(dir_path, output_dir, verbose=verbose)
                
                summary['total_patients'] += stats['patients']
                summary['total_volumes_created'] += stats['success']
                summary['total_volumes_failed'] += stats['failed']
                summary['total_files_skipped'] += stats['skipped']
                
                result = {
                    'directory': str(rel_path),
                    'input_path': str(dir_path),
                    'output_path': str(output_dir),
                    'patients': stats['patients'],
                    'slices': stats['slices'],
                    'volumes_created': stats['success'],
                    'volumes_failed': stats['failed'],
                    'files_skipped': stats['skipped']
                }
                summary['directory_results'].append(result)
                
                if verbose:
                    print(f"  [SUCCESS] Created {stats['success']} volumes from {stats['patients']} patients")
                    if stats['failed'] > 0:
                        print(f"  [WARN] Failed: {stats['failed']}")
                    if stats['skipped'] > 0:
                        print(f"  [WARN] Skipped {stats['skipped']} files (no patient ID)")
                
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] Failed to process directory: {e}")
                summary['directory_results'].append({
                    'directory': str(rel_path),
                    'error': str(e)
                })
        
        # Print final summary
        self._print_summary(summary)
        
        return summary
    
    @staticmethod
    def _print_summary(summary: Dict) -> None:
        """Print final processing summary"""
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Total directories processed: {summary['total_dirs']}")
        print(f"Total patients found: {summary['total_patients']}")
        print(f"Total volumes created: {summary['total_volumes_created']}")
        print(f"Total volumes failed: {summary['total_volumes_failed']}")
        if summary['total_files_skipped'] > 0:
            print(f"Total files skipped: {summary['total_files_skipped']} (no patient ID)")
        
        if summary['directory_results']:
            print(f"\nPer-directory breakdown:")
            for result in summary['directory_results']:
                if 'error' not in result:
                    print(f"  {result['directory']}: "
                          f"{result['volumes_created']} volumes from "
                          f"{result['patients']} patients ({result['slices']} slices)")
        
        print("="*70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch stack entire directory tree to patient-wise NIfTI volumes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and experiment specification (REQUIRED)
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help="Model name (e.g., 'fastflow', 'padim', 'patchcore')"
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=True,
        help="Experiment name (e.g., 'aggressive', 'conservative', 'baseline')"
    )
    
    # Base directories (with defaults)
    parser.add_argument(
        '--input-base',
        type=str,
        default="./masks_morpho",
        help="Input base directory (where morphology outputs are)"
    )
    
    parser.add_argument(
        '--output-base',
        type=str,
        default="./nifti_volumes",
        help="Output base directory for NIfTI volumes"
    )
    
    # NIfTI parameters
    parser.add_argument(
        '--spacing', '-s',
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=('X', 'Y', 'Z'),
        help="Voxel spacing in mm (x y z)"
    )
    
    parser.add_argument(
        '--origin',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=('X', 'Y', 'Z'),
        help="Volume origin in mm (x y z)"
    )
    
    parser.add_argument(
        '--patient-pattern',
        type=str,
        default=r"PA(\d+)_",
        help="Regex pattern to extract patient ID (must have one capture group)"
    )
    
    # Processing options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be processed without actually processing"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed progress for each directory"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build input and output paths
    input_base = Path(args.input_base) / f"{args.model}_{args.experiment}"
    output_base = Path(args.output_base) / f"{args.model}_{args.experiment}"
    
    # Validate input
    if not input_base.exists():
        print(f"[ERROR] Input directory not found: {input_base}")
        print(f"[INFO] Make sure you have run morphology processing first:")
        print(f"       python apply_morpho.py --model {args.model} --experiment {args.experiment}")
        sys.exit(1)
    
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Experiment: {args.experiment}")
    
    # Create stacker
    stacker = BatchNIfTIStacker(
        spacing=tuple(args.spacing),
        origin=tuple(args.origin),
        patient_pattern=args.patient_pattern
    )
    
    try:
        summary = stacker.process_tree(
            input_base=input_base,
            output_base=output_base,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        if summary['total_volumes_failed'] > 0:
            print("[WARN] Some volumes failed to create")
            sys.exit(1)
        
        if not args.dry_run:
            print(f"[SUCCESS] All volumes created")
            print(f"[OUTPUT] NIfTI volumes saved to: {output_base}")

        
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()