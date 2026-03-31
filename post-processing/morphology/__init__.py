"""
Morphology Post-processing Module

Core components for anomaly map post-processing:
- MorphologyProcessor: Single-mask processing with morphological operations
- BatchProcessor: Batch processing of multiple masks
"""

from .processor import MorphologyProcessor, BatchProcessor, ProcessingStats
from .slice_metrics import aggregate_slice_metrics, compute_slice_metrics

__all__ = [
    'MorphologyProcessor',
    'BatchProcessor',
    'ProcessingStats',
    'compute_slice_metrics',
    'aggregate_slice_metrics',
]
