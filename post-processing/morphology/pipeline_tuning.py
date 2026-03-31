"""
Validation Pipeline for Tuning 

Directly processes pre-computed masked prediction masks.


"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from PIL import Image

from postprocess_utils import apply_replacements, canonical_suffix, load_array
from processor import MorphologyProcessor, BatchProcessor
from slice_metrics import aggregate_slice_metrics, compute_slice_metrics


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
        self._initialise_evaluation()
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

    def _initialise_evaluation(self) -> None:
        """Prepare evaluation-related configuration."""
        evaluation_cfg = self.config.get('evaluation') or {}
        self._evaluation_config = evaluation_cfg
        self._evaluation_enabled = bool(evaluation_cfg.get('enabled', False))

        default_exts = evaluation_cfg.get('allowed_extensions', [".png", ".npy", ".npz"])
        self._ground_truth_extensions = {ext.lower() for ext in default_exts if isinstance(ext, str)}
        if not self._ground_truth_extensions:
            self._ground_truth_extensions = {".png", ".npy", ".npz"}

        self._metrics_prediction_threshold = evaluation_cfg.get('prediction_threshold')
        self._metrics_ground_truth_threshold = evaluation_cfg.get('ground_truth_threshold')
        self._metrics_skip_missing = bool(evaluation_cfg.get('skip_missing', True))
        self._metrics_store_per_slice = bool(evaluation_cfg.get('store_per_slice', False))

        self._ground_truth_replacements = self._parse_replacements(evaluation_cfg.get('replacements'))
        self._ground_truth_dir: Path | None = None

        if self._evaluation_enabled:
            ground_truth_dir = evaluation_cfg.get('ground_truth_dir')
            if ground_truth_dir is None:
                raise ValueError("Evaluation enabled but 'ground_truth_dir' is not specified in the config.")
            self._ground_truth_dir = Path(ground_truth_dir).expanduser().resolve()
            if not self._ground_truth_dir.exists():
                raise FileNotFoundError(f"Ground-truth directory not found: {self._ground_truth_dir}")

    @staticmethod
    def _parse_replacements(raw: Any) -> dict[str, str]:
        """Convert config replacements into a mapping."""
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        if isinstance(raw, str):
            if ":" not in raw:
                raise ValueError(f"Replacement '{raw}' must contain ':'")
            src, dst = raw.split(":", 1)
            return {src: dst}
        if isinstance(raw, (list, tuple)):
            mapping: dict[str, str] = {}
            for item in raw:
                mapping.update(MorphologyValidationPipelineSimplified._parse_replacements(item))
            return mapping
        raise TypeError(f"Unsupported replacement specification: {raw!r}")

    def _candidate_ground_truth_relatives(self, relative: Path) -> List[Path]:
        """Generate possible ground-truth filenames for a predicted slice."""
        replaced = apply_replacements(relative, self._ground_truth_replacements) if self._ground_truth_replacements else relative
        parent = replaced.parent
        stem = replaced.stem

        # Normalise common suffixes.
        base = stem
        for suffix in ("_pred_mask", "_anomaly_map"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        stem_variants = {
            stem,
            base,
            f"{base}_mask",
            f"{base}_label",
            f"{base}_gt",
        }

        alt_exts = set(self._ground_truth_extensions)
        canonical = canonical_suffix(replaced)
        if canonical:
            alt_exts.add(canonical)
        suffix = replaced.suffix.lower()
        if suffix:
            alt_exts.add(suffix)
        if not alt_exts:
            alt_exts.add(".png")

        candidates: List[Path] = []
        seen: set[Path] = set()
        for stem_variant in stem_variants:
            for ext in alt_exts:
                candidate = parent / f"{stem_variant}{ext}"
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
        return candidates

    def _resolve_ground_truth_path(self, relative: Path) -> tuple[Path | None, List[Path]]:
        """Return matching ground-truth path (if available) and the candidates considered."""
        if not self._ground_truth_dir:
            return None, []
        candidates_rel = self._candidate_ground_truth_relatives(relative)
        candidates: List[Path] = []
        for rel_candidate in candidates_rel:
            candidate = self._ground_truth_dir / rel_candidate
            candidates.append(candidate)
            if candidate.exists():
                return candidate, candidates
        return None, candidates

    @staticmethod
    def _load_mask_array(path: Path) -> np.ndarray:
        """Load prediction or ground-truth mask as a numpy array."""
        array_with_meta = load_array(path)
        return array_with_meta.data

    def _evaluate_experiment(self, output_dir: Path) -> dict[str, Any]:
        """Compute slice-level metrics for a processed experiment output."""
        prediction_files = sorted(
            [
                path
                for path in output_dir.glob("*")
                if path.is_file() and path.suffix.lower() in {".png", ".npy", ".npz"}
            ]
        )

        if not prediction_files:
            return {
                "overall": aggregate_slice_metrics([]),
                "evaluated_slices": 0,
                "missing_ground_truth": [],
                "per_slice": [] if self._metrics_store_per_slice else None,
            }

        metrics_accumulator: List[dict[str, float]] = []
        per_slice_records: List[dict[str, Any]] = []
        missing: List[str] = []

        for pred_path in prediction_files:
            relative = pred_path.relative_to(output_dir)
            ground_truth_path, checked_candidates = self._resolve_ground_truth_path(relative)
            if ground_truth_path is None:
                if self._metrics_skip_missing:
                    missing.append(str(relative))
                    continue
                preview = ", ".join(str(path) for path in checked_candidates[:5])
                raise FileNotFoundError(
                    f"Missing ground-truth for {relative} (searched {len(checked_candidates)} candidates; examples: {preview})"
                )

            prediction_array = self._load_mask_array(pred_path)
            ground_truth_array = self._load_mask_array(ground_truth_path)

            metrics = compute_slice_metrics(
                prediction_array,
                ground_truth_array,
                prediction_threshold=self._metrics_prediction_threshold,
                ground_truth_threshold=self._metrics_ground_truth_threshold,
            )
            metrics_accumulator.append(metrics)

            if self._metrics_store_per_slice:
                try:
                    gt_relative = ground_truth_path.relative_to(self._ground_truth_dir) if self._ground_truth_dir else ground_truth_path
                except ValueError:
                    gt_relative = ground_truth_path
                per_slice_records.append(
                    {
                        "relative_path": str(relative),
                        "ground_truth_relative": str(gt_relative),
                        **metrics,
                    }
                )

        overall = aggregate_slice_metrics(metrics_accumulator)
        return {
            "overall": overall,
            "evaluated_slices": overall.get("num_slices", len(metrics_accumulator)),
            "missing_ground_truth": missing,
            "per_slice": per_slice_records if self._metrics_store_per_slice else None,
        }
    
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
                arr = np.load(first_mask)
                unique = np.unique(arr)
            else:
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

            evaluation = None
            if self._evaluation_enabled:
                evaluation = self._evaluate_experiment(exp_output_dir)
            
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
                'evaluation': evaluation,
            }
            
            all_results.append(result)
            print(f"[RESULT] {exp_name}")
            print(f"  - Processed: {stats['processed']}")
            print(f"  - Failed: {stats['failed']}")
            print(f"  - Avg area preserved: {stats['avg_area_preserved']:.2%}")
            print(f"  - Components removed: {stats['total_components_removed_early']}")

            if evaluation is not None:
                if evaluation.get("evaluated_slices", 0) > 0 and evaluation.get("overall"):
                    overall = evaluation["overall"]
                    print(f"  - F1 score: {overall['f1_score']:.3f}")
                    print(f"  - Precision: {overall['precision']:.3f}")
                    print(f"  - Recall: {overall['recall']:.3f}")
                    print(f"  - False negative rate: {overall['false_negative_rate']:.3f}")
                    print(f"  - Balanced accuracy: {overall['balanced_accuracy']:.3f}")
                else:
                    print("  - Evaluation: no slices with matching ground truth.")
                missing_gt = evaluation.get("missing_ground_truth") or []
                if missing_gt:
                    print(f"  - Missing ground-truth slices: {len(missing_gt)}")
        
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
                f"{stats['total_components_removed_early']:>20}"
            )
            evaluation = result.get('evaluation')
            if evaluation and evaluation.get("evaluated_slices", 0) > 0 and evaluation.get("overall"):
                overall = evaluation["overall"]
                print(
                    f"    Metrics → "
                    f"F1:{overall['f1_score']:.3f} "
                    f"P:{overall['precision']:.3f} "
                    f"R:{overall['recall']:.3f} "
                    f"FNR:{overall['false_negative_rate']:.3f} "
                    f"BA:{overall['balanced_accuracy']:.3f}"
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
