"""
Evaluation Package for Consilium Fine-Tuning Pipeline

This package handles model evaluation and performance assessment.
"""

from .evaluator import ModelEvaluator
from .metrics import ConsensusMetrics

__all__ = [
    'ModelEvaluator',
    'ConsensusMetrics'
]
