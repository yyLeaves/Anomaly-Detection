"""
Validation Pipeline for Tuning 

Directly processes pre-computed masked prediction masks.


"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict
import yaml

from processor import MorphologyProcessor, BatchProcessor


class MorphologyValidationPipelineSimplified:
    """Simplified pipeline - starts directly from masked prediction masks"""
    
    def __init__(self, config_path: str | Path):
        """
        Initialize from YAML config
        
        Args:
            config_path: Path to morpho_validation.yaml
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
        self._prepare_directories()
    
    def _validate_config(self) -> None:
        """Validate essential config sections"""
        required_sections = ['paths', 'morphology', 'tuning_experiments']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def _prepare_directories(self) -> None:
        """Create necessary output directories"""
        output_base = Path(self.config['paths']['output_base'])
        viz_base = Path(self.config['paths']['visualization_base'])
        reports_base = Path(self.config['paths']['reports_base'])
        
        for directory in [output_base, viz_base, reports_base]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def verify_input_data(self) -> bool:
        """
        Verify that masked prediction masks already exist
        
        Returns:
            True if data exists and is valid
        """
        paths = self.config['paths']
        input_dir = Path(paths['output_base']) 
        
        if not input_dir.exists():
            print(f"[ERROR] Input directory not found: {input_dir}")
            return False
        
        mask_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.npy"))
        if not mask_files:
            print(f"[ERROR] No mask files found in {input_dir}")
            return False
        
        print(f"[OK] Found {len(mask_files)} mask files in {input_dir}")
        
        # Verify format of first mask
        first_mask = mask_files[0]
        try:
            if first_mask.suffix.lower() == '.npy':
                import numpy as np
                arr = np.load(first_mask)
                unique = np.unique(arr)
            else:
                from PIL import Image
                import numpy as np
                img = Image.open(first_mask)
                arr = np.array(img)
                unique = np.unique(arr)
            
            # Check if values are 0/1 or 0/255
            if set(unique) <= {0, 1}:
                print(f"[OK] Mask format: binary {0,1}")
            elif set(unique) <= {0, 255}:
                print(f"[OK] Mask format: binary {0,255}")
            else:
                print(f"[WARN] Mask format unexpected: unique values {unique}")
                print("       This might still work, but verify manually")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to verify mask format: {e}")
            return False
    
    def run_morphology_tuning(self) -> Dict:
        """
        Run morphology processing with multiple parameter combinations
        
        Returns:
            Dictionary with all tuning results
        """
        print("\n[PROCESSING] Running morphology parameter tuning...")
        print("="*70)
        
        paths = self.config['paths']
        input_dir = Path(paths['output_base']) 
        tuning_output_base = Path(paths['output_base']) / 'validation_morpho_tuning'
        
        all_results = []
        experiments = self.config['tuning_experiments']
        
        for idx, exp in enumerate(experiments):
            exp_name = exp['name']
            print(f"\n[EXPERIMENT {idx+1}/{len(experiments)}] {exp_name}")
            print("-"*70)
            
            # Create experiment output directory
            exp_output_dir = tuning_output_base / exp_name
            
            # Create processor with experiment parameters
            processor = MorphologyProcessor(
                dilate_iterations=exp.get('dilate_iterations', 1),
                erode_iterations=exp.get('erode_iterations', 1),
                num_rounds=exp.get('num_rounds', 1),
                kernel_size=exp.get('kernel_size', 5),
                kernel_shape=exp.get('kernel_shape', 'ellipse'),
                min_component_size=exp.get('min_component_size', 3),
                connectivity=exp.get('connectivity', 8),
            )
            
            batch_processor = BatchProcessor(processor)
            
            # Process directory
            stats = batch_processor.process_directory(
                input_dir=input_dir,
                output_dir=exp_output_dir,
                binarize_threshold=self.config['morphology']['binarize_threshold'],
                max_samples=self.config['debug'].get('max_samples'),
                verbose=self.config['debug'].get('verbose', True),
            )
            
            result = {
                'experiment_name': exp_name,
                'parameters': {
                    'dilate_iterations': exp.get('dilate_iterations', 1),
                    'erode_iterations': exp.get('erode_iterations', 1),
                    'num_rounds': exp.get('num_rounds', 1),
                    'kernel_size': exp.get('kernel_size', 5),
                    'min_component_size': exp.get('min_component_size', 3),
                },
                'statistics': stats,
                'output_dir': str(exp_output_dir),
            }
            
            all_results.append(result)
            print(f"[RESULT] {exp_name}")
            print(f"  - Processed: {stats['processed']}")
            print(f"  - Failed: {stats['failed']}")
            print(f"  - Avg area preserved: {stats['avg_area_preserved']:.2%}")
            print(f"  - Components removed: {stats['total_components_removed']}")
        
        return all_results
    
    def generate_reports(self, tuning_results: Dict) -> None:
        """
        Generate comparison reports
        
        Args:
            tuning_results: Results from run_morphology_tuning()
        """
        print("\n[REPORTING] Generating comparison reports...")
        print("="*70)
        
        reports_dir = Path(self.config['paths']['reports_base']) / 'morphology_tuning'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON report
        report_file = reports_dir / 'tuning_report.json'
        with open(report_file, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        
        print(f"[SUCCESS] Report saved: {report_file}")
        
        # Print summary table
        self._print_summary_table(tuning_results)
        
        # Print recommendations
        self._print_recommendations(tuning_results)
    
    @staticmethod
    def _print_summary_table(tuning_results: Dict) -> None:
        """Print results comparison table"""
        print("\n" + "="*100)
        print("Tuning Results Summary Table")
        print("="*100)
        print(f"{'Experiment':<40} {'Processed':>10} {'Area Preserved':>20} {'Removed Components':>20}")
        print("-"*100)
        
        for result in tuning_results:
            stats = result['statistics']
            print(
                f"{result['experiment_name']:<40} "
                f"{stats['processed']:>10} "
                f"{stats['avg_area_preserved']:>19.2%} "
                f"{stats['total_components_removed']:>20}"
            )
        
        print("="*100 + "\n")
    
    @staticmethod
    def _print_recommendations(tuning_results: Dict) -> None:
        """Print parameter recommendations based on results"""
        print("="*100)
        print("Parameter Recommendations")
        print("="*100)
        
        if not tuning_results:
            return
        
        # Best for preservation (Recall)
        best_preserve = max(
            tuning_results,
            key=lambda x: x['statistics']['avg_area_preserved']
        )
        print(f"\n1. Aggresive (Maximize Recall):")
        print(f"   Experiment: {best_preserve['experiment_name']}")
        print(f"   Parameters: {best_preserve['parameters']}")
        print(f"   Area Preserved: {best_preserve['statistics']['avg_area_preserved']:.2%}")
        print(f"   Try to minimize false negatives (all predictions kept)")
        
        # Best for removal (Precision)
        best_remove = min(
            tuning_results,
            key=lambda x: x['statistics']['avg_area_preserved']
        )
        print(f"\n2. Conservative (Maximize Precision):")
        print(f"   Experiment: {best_remove['experiment_name']}")
        print(f"   Parameters: {best_remove['parameters']}")
        print(f"   Area Preserved: {best_remove['statistics']['avg_area_preserved']:.2%}")
        print(f"   Try to minimize false positives (high confidence predictions)")
        
        # Balance
        balanced = min(
            tuning_results,
            key=lambda x: abs(x['statistics']['avg_area_preserved'] - 0.7)
        )
        print(f"\n3. Balanced (Compromise):")
        print(f"   Experiment: {balanced['experiment_name']}")
        print(f"   Parameters: {balanced['parameters']}")
        print(f"   Area Preserved: {balanced['statistics']['avg_area_preserved']:.2%}")
        
    
    
    def run(self) -> None:
        """Execute simplified validation pipeline"""
        print("\n" + "="*70)
        print("Morphology Validation Pipeline (Simplified)")
        print("Starting directly from masked prediction masks")
        print("="*70)
        print(f"Config: {self.config_path}")
        print("="*70)
        
        # Verify input data exists
        if not self.verify_input_data():
            print("\n[ERROR] Input data verification failed")
            sys.exit(1)
        
        print("\n[OK] Input data verified successfully\n")
        
        # Run morphology tuning
        tuning_results = self.run_morphology_tuning()
        
        # Generate reports
        self.generate_reports(tuning_results)
        
        print("\n[SUCCESS] Pipeline completed successfully!")
        print(f"Results saved to: {self.config['paths']['output_base']}/validation_morpho_tuning/")
        print(f"Reports saved to: {self.config['paths']['reports_base']}/morphology_tuning/")