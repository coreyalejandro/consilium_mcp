"""
Data Collection Package for Consilium Fine-Tuning Pipeline

This package handles the collection, processing, and quality assessment
of training data from AI consensus discussions.
"""

from .dataset_logger import log_training_example
from .quality_scorer import QualityScorer
from .data_processor import DataProcessor

__all__ = [
    'log_training_example',
    'QualityScorer', 
    'DataProcessor'
]
