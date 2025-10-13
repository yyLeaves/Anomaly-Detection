"""
Morphology Post-processing Module

Core components for FastFlow anomaly map post-processing:
- MorphologyProcessor: Single-mask processing with morphological operations
- BatchProcessor: Batch processing of multiple masks
"""

from .processor import MorphologyProcessor, BatchProcessor, ProcessingStats

__all__ = [
    'MorphologyProcessor',
    'BatchProcessor',
    'ProcessingStats',
]