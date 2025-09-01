"""
Universal Fine-Tuning Pipeline

Main application for the fine-tuning pipeline that provides
a Gradio interface for data collection, processing, and training
for ANY AI model.
"""

import gradio as gr
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import our modules
from data_collection import DataProcessor, QualityScorer
from training import TrainingConfig
from evaluation import ModelEvaluator

# Load environment variables
load_dotenv()

class FineTuningPipeline:
    """Main fine-tuning pipeline application"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.quality_scorer = QualityScorer()
        self.evaluator = ModelEvaluator()
        
    def process_training_data(self, 
                            input_file: str,
                            quality_threshold: float,
                            train_ratio: float,
                            val_ratio: float,
                            test_ratio: float) -> Dict[str, Any]:
        """Process raw training data"""
        
        try:
            # Load data
            examples = self.data_processor.load_training_data(input_file)
            
            # Filter and split
            train_data, val_data, test_data = self.data_processor.filter_and_split(
                examples, train_ratio, val_ratio, test_ratio
            )
            
            # Generate statistics
            stats = self.data_processor.generate_statistics(examples)
            
            # Save processed data
            self.data_processor.save_processed_data(
                train_data, "data/processed/train.jsonl", "sft"
            )
            self.data_processor.save_processed_data(
                val_data, "data/processed/val.jsonl", "sft"
            )
            self.data_processor.save_processed_data(
                test_data, "data/processed/test.jsonl", "sft"
            )
            
            return {
                "status": "✅ Success",
                "total_examples": stats["total_examples"],
                "train_examples": len(train_data),
                "val_examples": len(val_data),
                "test_examples": len(test_data),
                "avg_quality": stats.get("quality_distribution", {}).get("mean", 0),
                "statistics": json.dumps(stats, indent=2)
            }
            
        except Exception as e:
            return {
                "status": f"❌ Error: {str(e)}",
                "total_examples": 0,
                "train_examples": 0,
                "val_examples": 0,
                "test_examples": 0,
                "avg_quality": 0,
                "statistics": ""
            }
    
    def start_training(self,
                      base_model: str,
                      learning_rate: float,
                      batch_size: int,
                      epochs: int,
                      use_lora: bool,
                      use_wandb: bool) -> Dict[str, Any]:
        """Start the fine-tuning process"""
        
        try:
            # Create training config
            config = TrainingConfig(
                base_model=base_model,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=epochs,
                use_lora=use_lora,
                use_wandb=use_wandb
            )
            
            # Validate config
            config.validate()
            
            # Save config
            config.save("configs/training_config.json")
            
            return {
                "status": "✅ Training started",
                "config": json.dumps(config.to_dict(), indent=2),
                "model_path": config.output_dir
            }
            
        except Exception as e:
            return {
                "status": f"❌ Error: {str(e)}",
                "config": "",
                "model_path": ""
            }
    
    def evaluate_model(self, model_path: str, test_data_path: str) -> Dict[str, Any]:
        """Evaluate a fine-tuned model"""
        
        try:
            # Load test data
            test_examples = self.data_processor.load_training_data(test_data_path)
            
            # Run evaluation
            results = self.evaluator.evaluate_model(model_path, test_examples)
            
            return {
                "status": "✅ Evaluation complete",
                "consensus_quality": results.get("consensus_quality", 0),
                "research_depth": results.get("research_depth", 0),
                "reasoning_clarity": results.get("reasoning_clarity", 0),
                "overall_score": results.get("overall_score", 0),
                "detailed_results": json.dumps(results, indent=2)
            }
            
        except Exception as e:
            return {
                "status": f"❌ Error: {str(e)}",
                "consensus_quality": 0,
                "research_depth": 0,
                "reasoning_clarity": 0,
                "overall_score": 0,
                "detailed_results": ""
            }

def create_interface():
    """Create the Gradio interface"""
    
    pipeline = FineTuningPipeline()
    
    with gr.Blocks(title="Consilium Fine-Tuning Pipeline", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🎯 Consilium Fine-Tuning Pipeline
        
        Advanced fine-tuning pipeline for improving multi-AI consensus models through data collection, training, and evaluation.
        """)
        
        with gr.Tabs():
            
            # Data Processing Tab
            with gr.Tab("📊 Data Processing"):
                gr.Markdown("""
                ## Process Training Data
                
                Load raw training data from Consilium MCP discussions and prepare it for fine-tuning.
                """)
                
                with gr.Row():
                    with gr.Column():
                        input_file = gr.File(
                            label="Training Data File (JSONL)",
                            file_types=[".jsonl"],
                            value="data/training.jsonl"
                        )
                        
                        quality_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                            label="Quality Threshold",
                            info="Minimum quality score for examples"
                        )
                        
                        with gr.Row():
                            train_ratio = gr.Slider(
                                minimum=0.5, maximum=0.9, value=0.8, step=0.1,
                                label="Train Ratio"
                            )
                            val_ratio = gr.Slider(
                                minimum=0.05, maximum=0.3, value=0.1, step=0.05,
                                label="Validation Ratio"
                            )
                            test_ratio = gr.Slider(
                                minimum=0.05, maximum=0.3, value=0.1, step=0.05,
                                label="Test Ratio"
                            )
                        
                        process_btn = gr.Button("🔄 Process Data", variant="primary")
                    
                    with gr.Column():
                        status_output = gr.Textbox(label="Status", interactive=False)
                        stats_output = gr.JSON(label="Statistics")
            
            # Training Tab
            with gr.Tab("🚀 Training"):
                gr.Markdown("""
                ## Fine-Tune Model
                
                Configure and start the fine-tuning process.
                """)
                
                with gr.Row():
                    with gr.Column():
                        base_model = gr.Dropdown(
                            choices=[
                                "mistralai/Mistral-7B-v0.1",
                                "microsoft/DialoGPT-medium",
                                "gpt2",
                                "custom"
                            ],
                            value="mistralai/Mistral-7B-v0.1",
                            label="Base Model"
                        )
                        
                        learning_rate = gr.Slider(
                            minimum=1e-6, maximum=1e-3, value=2e-5, step=1e-6,
                            label="Learning Rate"
                        )
                        
                        batch_size = gr.Slider(
                            minimum=1, maximum=16, value=4, step=1,
                            label="Batch Size"
                        )
                        
                        epochs = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Epochs"
                        )
                        
                        with gr.Row():
                            use_lora = gr.Checkbox(label="Use LoRA", value=True)
                            use_wandb = gr.Checkbox(label="Use Weights & Biases", value=True)
                        
                        train_btn = gr.Button("🚀 Start Training", variant="primary")
                    
                    with gr.Column():
                        training_status = gr.Textbox(label="Training Status", interactive=False)
                        config_output = gr.JSON(label="Configuration")
            
            # Evaluation Tab
            with gr.Tab("📈 Evaluation"):
                gr.Markdown("""
                ## Evaluate Model
                
                Assess the performance of fine-tuned models.
                """)
                
                with gr.Row():
                    with gr.Column():
                        model_path = gr.Textbox(
                            label="Model Path",
                            value="models/finetuned",
                            placeholder="Path to fine-tuned model"
                        )
                        
                        test_data_path = gr.Textbox(
                            label="Test Data Path",
                            value="data/processed/test.jsonl",
                            placeholder="Path to test data"
                        )
                        
                        eval_btn = gr.Button("📊 Evaluate Model", variant="primary")
                    
                    with gr.Column():
                        eval_status = gr.Textbox(label="Evaluation Status", interactive=False)
                        eval_results = gr.JSON(label="Results")
            
            # Configuration Tab
            with gr.Tab("⚙️ Configuration"):
                gr.Markdown("""
                ## Pipeline Configuration
                
                Manage environment variables and settings.
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Environment Variables")
                        
                        mistral_key = gr.Textbox(
                            label="Mistral API Key",
                            type="password",
                            value=os.getenv("MISTRAL_API_KEY", "")
                        )
                        
                        sambanova_key = gr.Textbox(
                            label="SambaNova API Key", 
                            type="password",
                            value=os.getenv("SAMBANOVA_API_KEY", "")
                        )
                        
                        openai_key = gr.Textbox(
                            label="OpenAI API Key",
                            type="password", 
                            value=os.getenv("OPENAI_API_KEY", "")
                        )
                        
                        save_config_btn = gr.Button("💾 Save Configuration")
                    
                    with gr.Column():
                        gr.Markdown("### Current Settings")
                        
                        current_config = gr.JSON(
                            label="Environment Configuration",
                            value={
                                "USE_DATASET_LOGGING": os.getenv("USE_DATASET_LOGGING", "0"),
                                "DATASET_LOG_PATH": os.getenv("DATASET_LOG_PATH", "data/training.jsonl"),
                                "QUALITY_THRESHOLD": os.getenv("QUALITY_THRESHOLD", "0.7"),
                                "BASE_MODEL": os.getenv("BASE_MODEL", "mistralai/Mistral-7B-v0.1"),
                                "LEARNING_RATE": os.getenv("LEARNING_RATE", "2e-5"),
                                "BATCH_SIZE": os.getenv("BATCH_SIZE", "4"),
                                "EPOCHS": os.getenv("EPOCHS", "3")
                            }
                        )
        
        # Event handlers
        process_btn.click(
            pipeline.process_training_data,
            inputs=[input_file, quality_threshold, train_ratio, val_ratio, test_ratio],
            outputs=[status_output, stats_output]
        )
        
        train_btn.click(
            pipeline.start_training,
            inputs=[base_model, learning_rate, batch_size, epochs, use_lora, use_wandb],
            outputs=[training_status, config_output]
        )
        
        eval_btn.click(
            pipeline.evaluate_model,
            inputs=[model_path, test_data_path],
            outputs=[eval_status, eval_results]
        )
    
    return demo

if __name__ == "__main__":
    # Create necessary directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models/finetuned").mkdir(parents=True, exist_ok=True)
    Path("configs").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    # Launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        debug=False
    )