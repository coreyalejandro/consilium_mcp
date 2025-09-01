"""
Training Configuration for Fine-Tuning Pipeline

Manages training parameters and configurations for different
fine-tuning approaches.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning training"""
    
    # Model configuration
    base_model: str = "mistralai/Mistral-7B-v0.1"
    model_name: str = "consilium-finetuned"
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # Data configuration
    train_data_path: str = "data/processed/train.jsonl"
    val_data_path: str = "data/processed/val.jsonl"
    test_data_path: str = "data/processed/test.jsonl"
    
    # Output configuration
    output_dir: str = "models/finetuned"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Optimization
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    fp16: bool = True
    bf16: bool = False
    
    # LoRA configuration (if using PEFT)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # DPO specific
    beta: float = 0.1  # DPO beta parameter
    
    # Evaluation
    evaluation_metrics: list = field(default_factory=lambda: [
        "consensus_quality", "research_depth", "reasoning_clarity"
    ])
    
    # Logging and monitoring
    use_wandb: bool = True
    wandb_project: str = "consilium-finetune"
    log_level: str = "INFO"
    
    # Advanced settings
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = True
    
    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Create config from environment variables"""
        return cls(
            base_model=os.getenv("BASE_MODEL", "mistralai/Mistral-7B-v0.1"),
            learning_rate=float(os.getenv("LEARNING_RATE", "2e-5")),
            batch_size=int(os.getenv("BATCH_SIZE", "4")),
            gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")),
            max_length=int(os.getenv("MAX_LENGTH", "2048")),
            num_epochs=int(os.getenv("EPOCHS", "3")),
            use_lora=os.getenv("USE_LORA", "1") == "1",
            use_wandb=os.getenv("USE_WANDB", "1") == "1",
            wandb_project=os.getenv("WANDB_PROJECT", "consilium-finetune")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "base_model": self.base_model,
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_clipping": self.gradient_clipping,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "beta": self.beta,
            "seed": self.seed
        }
    
    def save(self, path: str) -> None:
        """Save config to file"""
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        
        if self.max_length <= 0:
            raise ValueError("Max length must be positive")
        
        if self.use_lora and (self.lora_r <= 0 or self.lora_alpha <= 0):
            raise ValueError("LoRA parameters must be positive")
        
        # Check data paths exist
        for path_name, path in [
            ("train_data_path", self.train_data_path),
            ("val_data_path", self.val_data_path),
            ("test_data_path", self.test_data_path)
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"{path_name} does not exist: {path}")
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get training arguments for transformers"""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.gradient_clipping,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "remove_unused_columns": self.remove_unused_columns,
            "dataloader_num_workers": self.dataloader_num_workers,
            "seed": self.seed,
            "report_to": ["wandb"] if self.use_wandb else None,
            "run_name": self.model_name
        }
