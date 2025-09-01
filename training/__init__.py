"""
Training Package for Consilium Fine-Tuning Pipeline

This package handles the fine-tuning process, including
SFT, DPO, and instruction tuning approaches.
"""

from .trainer import FineTuningTrainer
from .config import TrainingConfig
from .dspy_optimizer import DSPyOptimizer

__all__ = [
    'FineTuningTrainer',
    'TrainingConfig', 
    'DSPyOptimizer'
]
