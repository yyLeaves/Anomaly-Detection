#!/usr/bin/env python3
"""
Apply Selected Morphology Parameters to Entire Model Directory

After tuning on validation set, use this script to apply the optimal parameters
to all splits (train/valid/test) and categories (Good/Ungood) in any model directory.

Usage Examples:
    # Apply aggressive parameters to FastFlow model
    python apply_morpho.py \
      --model fastflow \
      --experiment aggressive \
      --config config/morpho_validation.yaml
    
    # Apply to different model with conservative parameters
    python apply_morpho.py \
      --model padim \
      --experiment conservative \
      --config config/morpho_validation.yaml
    
    # Custom base directories
    python apply_morpho.py \
      --model fastflow \
      --experiment aggressive \
      --config config/morpho_validation.yaml \
      --input-base ./masks_ab \
      --output-base ./masks_morpho
    
    # Specify parameters directly (no config needed)
    python apply_morpho.py \
      --model fastflow \
      --dilate 2 --erode 1 --rounds 1 --kernel 5 --min-size 3

Directory structure:
    Input:  {input_base}/{model}/valid/Ungood/*.png
            {input_base}/{model}/test/Ungood/*.png
           
    Output: {output_base}/{model}_{experiment}/valid/Ungood/*.png
            {output_base}/{model}_{experiment}/test/Ungood/*.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from processor import MorphologyProcessor, BatchProcessor


class BatchMorphologyApplicator:
    """Apply morphology parameters to entire directory tree"""
    
    def __init__(self, processor: MorphologyProcessor):
        """
        Initialize applicator
        
        Args:
            processor: Configured MorphologyProcessor with desired parameters
        """
        self.processor = processor
        self.batch_processor = BatchProcessor(processor)
    
    def find_mask_directories(self, base_dir: Path) -> list[Path]:
        """
        Find all directories containing PNG mask files
        
        Searches recursively for directories with *.png files
        
        Args:
            base_dir: Root directory to search
            
        Returns:
            List of directories containing PNG files
        """
        mask_dirs = []
        
        for path in base_dir.rglob("*"):
            if path.is_dir():
                # Check if directory contains PNG files
                png_files = list(path.glob("*.png"))
                if png_files:
                    mask_dirs.append(path)
        
        return sorted(mask_dirs)
    
    def get_relative_path(self, full_path: Path, base_dir: Path) -> Path:
        """Get relative path from base directory"""
        try:
            return full_path.relative_to(base_dir)
        except ValueError:
            # If not relative, return the full path
            return full_path
    
    def process_tree(
        self,
        input_base: Path,
        output_base: Path,
        binarize_threshold: float = 0.5,
        dry_run: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Process entire directory tree
        
        Args:
            input_base: Input base directory
            output_base: Output base directory
            binarize_threshold: Threshold for binarization
            dry_run: If True, only show what would be processed
            verbose: Print progress
            
        Returns:
            Summary statistics
        """
        print("\n" + "="*70)
        print("Batch Morphology Application")
        print("="*70)
        print(f"Input:  {input_base}")
        print(f"Output: {output_base}")
        print(f"\nMorphology Parameters:")
        print(f"  - Dilate iterations: {self.processor.dilate_iterations}")
        print(f"  - Erode iterations: {self.processor.erode_iterations}")
        print(f"  - Num rounds: {self.processor.num_rounds}")
        print(f"  - Kernel size: {self.processor.kernel_size} ({self.processor.kernel_shape})")
        print(f"  - Min component size: {self.processor.min_component_size} pixels")
        print(f"  - Connectivity: {self.processor.connectivity}")
        print("="*70 + "\n")
        
        # Find all directories with masks
        mask_dirs = self.find_mask_directories(input_base)
        
        if not mask_dirs:
            print(f"[ERROR] No directories with PNG files found in {input_base}")
            return {'total_dirs': 0, 'processed_dirs': 0, 'failed_dirs': 0}
        
        print(f"[INFO] Found {len(mask_dirs)} directories with PNG files:\n")
        for dir_path in mask_dirs:
            rel_path = self.get_relative_path(dir_path, input_base)
            num_files = len(list(dir_path.glob("*.png")))
            print(f"  - {rel_path}/ ({num_files} files)")
        
        if dry_run:
            print("\n[DRY RUN] No processing performed. Remove --dry-run to process.")
            return {'total_dirs': len(mask_dirs), 'processed_dirs': 0, 'failed_dirs': 0}
        
        print("\n" + "="*70)
        print("Processing...")
        print("="*70 + "\n")
        
        # Process each directory
        summary = {
            'total_dirs': len(mask_dirs),
            'processed_dirs': 0,
            'failed_dirs': 0,
            'total_files_processed': 0,
            'total_files_failed': 0,
            'directory_results': []
        }
        
        for dir_path in mask_dirs:
            rel_path = self.get_relative_path(dir_path, input_base)
            output_dir = output_base / rel_path
            
            print(f"\n[PROCESSING] {rel_path}/")
            print("-" * 70)
            
            try:
                stats = self.batch_processor.process_directory(
                    input_dir=dir_path,
                    output_dir=output_dir,
                    binarize_threshold=binarize_threshold,
                    verbose=False  # Suppress per-file output
                )
                
                summary['processed_dirs'] += 1
                summary['total_files_processed'] += stats['processed']
                summary['total_files_failed'] += stats['failed']
                
                result = {
                    'directory': str(rel_path),
                    'input_path': str(dir_path),
                    'output_path': str(output_dir),
                    'files_processed': stats['processed'],
                    'files_failed': stats['failed'],
                    'avg_area_preserved': stats['avg_area_preserved'],
                    'components_removed': stats['total_components_removed_early']
                }
                summary['directory_results'].append(result)
                
                print(f"[SUCCESS] Processed {stats['processed']} files")
                print(f"  - Failed: {stats['failed']}")
                if stats['failed'] > 0:
                    print(f"  - ERROR DETAILS:")
                    # Show first 3 errors
                    for i, fail_info in enumerate(stats['failed_files'][:3]):
                        print(f"    {fail_info['file']}: {fail_info['error']}")
                    if len(stats['failed_files']) > 3:
                        print(f"    ... and {len(stats['failed_files']) - 3} more")
                print(f"  - Avg area preserved: {stats['avg_area_preserved']:.2%}")
                print(f"  - Components removed: {stats['total_components_removed_early']}")
                
            except Exception as e:
                summary['failed_dirs'] += 1
                print(f"[ERROR] Failed to process directory: {e}")
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
        print(f"Total directories: {summary['total_dirs']}")
        print(f"Successfully processed: {summary['processed_dirs']}")
        print(f"Failed: {summary['failed_dirs']}")
        print(f"\nTotal files processed: {summary['total_files_processed']}")
        print(f"Total files failed: {summary['total_files_failed']}")
        
        if summary['directory_results']:
            successful = [r for r in summary['directory_results'] if 'error' not in r]
            if successful:
                avg_preserved = sum(r['avg_area_preserved'] for r in successful) / len(successful)
                total_components = sum(r['components_removed'] for r in successful)
                print(f"\nOverall average area preserved: {avg_preserved:.2%}")
                print(f"Total components removed: {total_components}")
        
        print("="*70 + "\n")


def load_experiment_params(config_path: Path, experiment_name: str) -> Dict:
    """
    Load parameters from config file for a specific experiment
    
    Args:
        config_path: Path to YAML config file
        experiment_name: Name of experiment (e.g., "aggressive")
        
    Returns:
        Dictionary of parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiments = config.get('tuning_experiments', [])
    
    for exp in experiments:
        if exp['name'] == experiment_name:
            return {
                'dilate_iterations': exp.get('dilate_iterations', 1),
                'erode_iterations': exp.get('erode_iterations', 1),
                'num_rounds': exp.get('num_rounds', 1),
                'kernel_size': exp.get('kernel_size', 5),
                'kernel_shape': exp.get('kernel_shape', 'ellipse'),
                'min_component_size': exp.get('min_component_size', 3),
                'connectivity': exp.get('connectivity', 8),
            }
    
    raise ValueError(f"Experiment '{experiment_name}' not found in config. Available: {[e['name'] for e in experiments]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply selected morphology parameters to entire model directory tree",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model specification (REQUIRED)
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help="Model name (e.g., 'fastflow', 'padim', 'patchcore')"
    )
    
    # Base directories (with defaults)
    parser.add_argument(
        '--input-base',
        type=str,
        default="./masks_ab",
        help="Input base directory"
    )
    
    parser.add_argument(
        '--output-base',
        type=str,
        default="./masks_morpho",
        help="Output base directory"
    )
    
    # Option 1: Load from config (RECOMMENDED)
    parser.add_argument(
        '--config', '-c',
        type=str,
        help="Path to config YAML file (e.g., config/morpho_validation.yaml)"
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        help="Experiment name from config (e.g., 'aggressive', 'conservative', 'baseline')"
    )
    
    # Option 2: Specify parameters directly
    parser.add_argument(
        '--dilate',
        type=int,
        help="Dilation iterations"
    )
    
    parser.add_argument(
        '--erode',
        type=int,
        help="Erosion iterations"
    )
    
    parser.add_argument(
        '--rounds',
        type=int,
        help="Number of morphology rounds"
    )
    
    parser.add_argument(
        '--kernel',
        type=int,
        help="Kernel size (must be odd)"
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        help="Minimum component size (pixels)"
    )
    
    parser.add_argument(
        '--kernel-shape',
        type=str,
        choices=['ellipse', 'rect'],
        default='ellipse',
        help="Kernel shape"
    )
    
    parser.add_argument(
        '--connectivity',
        type=int,
        choices=[4, 8],
        default=8,
        help="Connectivity for component filtering"
    )
    
    # Other options
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help="Binarization threshold"
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be processed without actually processing"
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Print detailed progress"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build input and output paths
    input_base = Path(args.input_base) / args.model
    
    # Determine experiment name for output directory
    if args.experiment:
        experiment_name = args.experiment
        output_base = Path(args.output_base) / f"{args.model}_{experiment_name}"
    else:
        experiment_name = "custom"
        output_base = Path(args.output_base) / f"{args.model}_custom"
    
    # Validate input
    if not input_base.exists():
        print(f"[ERROR] Input directory not found: {input_base}")
        print(f"[INFO] Make sure the model name is correct and the directory exists")
        sys.exit(1)
    
    # Determine parameters
    if args.config and args.experiment:
        # Load from config (RECOMMENDED)
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
        
        print(f"[INFO] Model: {args.model}")
        print(f"[INFO] Loading parameters from: {config_path}")
        print(f"[INFO] Using experiment: {args.experiment}\n")
        
        try:
            params = load_experiment_params(config_path, args.experiment)
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
    
    elif all([args.dilate is not None, args.erode is not None, 
              args.rounds is not None, args.kernel is not None, 
              args.min_size is not None]):
        # Use command-line parameters
        print(f"[INFO] Model: {args.model}")
        print("[INFO] Using parameters from command line\n")
        params = {
            'dilate_iterations': args.dilate,
            'erode_iterations': args.erode,
            'num_rounds': args.rounds,
            'kernel_size': args.kernel,
            'kernel_shape': args.kernel_shape,
            'min_component_size': args.min_size,
            'connectivity': args.connectivity,
        }
    
    else:
        print("[ERROR] Must specify parameters using one of:")
        print("  1. --config <path> --experiment <name>  (RECOMMENDED)")
        print("  2. All of: --dilate, --erode, --rounds, --kernel, --min-size")
        print("\nExample:")
        print(f"  python {sys.argv[0]} --model fastflow --config config/morpho_validation.yaml --experiment aggressive")
        sys.exit(1)
    
    # Create processor
    try:
        processor = MorphologyProcessor(**params)
    except ValueError as e:
        print(f"[ERROR] Invalid parameters: {e}")
        sys.exit(1)
    
    # Create applicator and process
    applicator = BatchMorphologyApplicator(processor)
    
    try:
        summary = applicator.process_tree(
            input_base=input_base,
            output_base=output_base,
            binarize_threshold=args.threshold,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        if summary['failed_dirs'] > 0:
            print("[WARN] Some directories failed to process")
            sys.exit(1)
        
        if not args.dry_run:
            print(f"[SUCCESS] All processing completed")
            print(f"[OUTPUT] Results saved to: {output_base}")
        
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()